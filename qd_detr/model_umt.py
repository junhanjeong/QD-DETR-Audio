"""
QD-DETR model with UMT-style video-audio fusion
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from qd_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx
from qd_detr.matcher import build_matcher
from qd_detr.transformer import build_transformer
from qd_detr.position_encoding import build_position_encoding
from qd_detr.misc import accuracy
from qd_detr.model import MLP, LinearLayer, SetCriterion  # Import original components
from qd_detr.umt_components import UniModalEncoder, CrossModalEncoder

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class QDDETR_UMT(nn.Module):
    """ QD DETR with UMT-style video-audio fusion. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, 
                 aud_dim=0, umt_hidden_dim=256, umt_num_tokens=4, umt_num_layers=1):
        """ 
        Initializes the UMT-enhanced QD-DETR model.
        
        Parameters:
            transformer: torch module of the transformer architecture
            position_embed: torch module of the position_embedding
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries
            aux_loss: True if auxiliary decoding losses are to be used
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting embeddings before contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
            umt_hidden_dim: hidden dimension for UMT components
            umt_num_tokens: number of bottleneck tokens
            umt_num_layers: number of UMT layers
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, 2)
        
        # UMT components
        self.umt_hidden_dim = umt_hidden_dim
        self.video_encoder = UniModalEncoder(vid_dim, umt_hidden_dim)
        self.audio_encoder = UniModalEncoder(aud_dim, umt_hidden_dim)
        self.cross_modal_encoder = CrossModalEncoder(
            umt_hidden_dim, 
            num_tokens=umt_num_tokens, 
            num_layers=umt_num_layers
        )
        
        # Project UMT output to transformer hidden dimension
        self.umt_to_transformer = nn.Linear(umt_hidden_dim, hidden_dim)
        
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        
        # Remove the old video projection since we're using UMT
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.aux_loss = aux_loss

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

    def forward(self, src_txt, src_txt_mask, src_vid=None, src_vid_mask=None, src_aud=None, src_aud_mask=None):
        """
        Forward pass with UMT-style video-audio fusion
        
        Args:
            src_txt: [batch_size, L_txt, D_txt] - text features
            src_txt_mask: [batch_size, L_txt] - text mask (0 on padded)
            src_vid: [batch_size, L_vid, D_vid] - video features
            src_vid_mask: [batch_size, L_vid] - video mask
            src_aud: [batch_size, L_aud, D_aud] - audio features  
            src_aud_mask: [batch_size, L_aud] - audio mask
        """
        if src_vid is None or src_aud is None:
            raise ValueError("Both src_vid and src_aud must be provided for UMT fusion")
            
        # Step 1: Encode video and audio separately using UniModalEncoders
        vid_encoded = self.video_encoder(src_vid, src_vid_mask)  # [batch_size, L_vid, umt_hidden_dim]
        aud_encoded = self.audio_encoder(src_aud, src_aud_mask)  # [batch_size, L_aud, umt_hidden_dim]
        
        # Step 2: Fuse video and audio using CrossModalEncoder  
        fused_features = self.cross_modal_encoder(aud_encoded, vid_encoded, mask=src_vid_mask)  # [batch_size, umt_hidden_dim]
        
        # Step 3: Project fused features to match video sequence length
        # Expand fused features to match video length for compatibility with existing architecture
        batch_size, vid_len = src_vid.shape[0], src_vid.shape[1]
        src_vid_fused = fused_features.unsqueeze(1).expand(-1, vid_len, -1)  # [batch_size, L_vid, umt_hidden_dim]
        
        # Project to transformer hidden dimension
        src_vid_final = self.umt_to_transformer(src_vid_fused)  # [batch_size, L_vid, hidden_dim]
        
        # Step 4: Process text (same as original)
        src_txt = self.input_txt_proj(src_txt)
        
        # Step 5: Concatenate video and text features
        src = torch.cat([src_vid_final, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        
        # Positional embeddings
        pos_vid = self.position_embed(src_vid_final, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        
        # Add global token (same as original)
        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        video_length = src_vid_final.shape[1]
        
        # Step 6: Apply transformer (same as original)
        hs, reference, memory, memory_global = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length)
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        txt_mem = memory[:, src_vid_final.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid_final.shape[1]]  # (bsz, L_vid, d)
        
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))
            
        # Check for empty text query
        if src_txt.shape[1] == 0:
            print("There is zero text query. You should change codes properly")
            exit(-1)

        ### Neg Pairs ###
        src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
        src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
        src_neg = torch.cat([src_vid_final, src_txt_neg], dim=1)
        mask_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()

        mask_neg = torch.cat([mask_, mask_neg], dim=1)
        src_neg = torch.cat([src_, src_neg], dim=1)
        pos_neg = pos.clone()  # since it does not use actual content

        _, _, memory_neg, memory_global_neg = self.transformer(src_neg, ~mask_neg, self.query_embed.weight, pos_neg, video_length=video_length)
        vid_mem_neg = memory_neg[:, :src_vid_final.shape[1]]

        out["saliency_scores"] = (torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        out["video_mask"] = src_vid_mask
        
        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        return out


def build_qddetr_umt(args):
    """
    Build the UMT-enhanced QD-DETR model
    """
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = QDDETR_UMT(
        transformer=transformer,
        position_embed=position_embedding,
        txt_position_embed=txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim,
        max_v_l=args.max_v_l,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
        aud_dim=args.a_feat_dim,
        umt_hidden_dim=getattr(args, 'umt_hidden_dim', 256),
        umt_num_tokens=getattr(args, 'umt_num_tokens', 4),
        umt_num_layers=getattr(args, 'umt_num_layers', 1)
    )

    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_span'] = args.span_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_label'] = args.label_loss_coef
    weight_dict['loss_saliency'] = args.lw_saliency
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
        weight_dict['loss_contrastive_align'] = args.contrastive_align_loss_coef

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        temperature=args.temperature,
        span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin
    )
    criterion.to(device)
    return model, criterion
