"""
UMT-specific dataset class for QD-DETR-Audio
This dataset ensures both video and audio features are loaded for UMT fusion
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from qd_detr.span_utils import span_xx_to_cxw
from qd_detr.start_end_dataset_audio import StartEndDataset_audio

logger = logging.getLogger(__name__)


class StartEndDataset_UMT(StartEndDataset_audio):
    """
    UMT-specific dataset that ensures both video and audio features are available
    for proper UMT fusion.
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir, a_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 dset_domain=None):
        
        # Ensure both video and audio are required for UMT
        if not a_feat_dir:
            raise ValueError("Audio feature directory (a_feat_dir) is required for UMT fusion")
        if not v_feat_dirs or (isinstance(v_feat_dirs, list) and not any(v_feat_dirs)):
            raise ValueError("Video feature directories (v_feat_dirs) are required for UMT fusion")
        
        super().__init__(
            dset_name=dset_name,
            data_path=data_path,
            v_feat_dirs=v_feat_dirs,
            q_feat_dir=q_feat_dir,
            a_feat_dir=a_feat_dir,
            q_feat_type=q_feat_type,
            max_q_l=max_q_l,
            max_v_l=max_v_l,
            data_ratio=data_ratio,
            ctx_mode=ctx_mode,
            normalize_v=normalize_v,
            normalize_t=normalize_t,
            load_labels=load_labels,
            clip_len=clip_len,
            max_windows=max_windows,
            span_loss_type=span_loss_type,
            txt_drop_ratio=txt_drop_ratio,
            dset_domain=dset_domain
        )
        
        # Force both video and audio usage for UMT
        self.use_video = True
        self.use_audio = True
        
        # Set feature dimensions - these will be computed from first loaded feature
        self.v_feat_dim = None
        self.a_feat_dim = None
        self._compute_feature_dims()
        
        logger.info(f"UMT Dataset initialized with {len(self.data)} examples")
        logger.info(f"Video dirs: {self.v_feat_dirs}")
        logger.info(f"Audio dir: {self.a_feat_dir}")
        logger.info(f"Video feature dim: {self.v_feat_dim}")
        logger.info(f"Audio feature dim: {self.a_feat_dim}")

    def _compute_feature_dims(self):
        """Compute feature dimensions from the first available feature file"""
        # Find first data sample to get feature dimensions
        for meta in self.data:
            vid_name = meta["vid"]
            
            # Try to get video feature dimension
            if self.v_feat_dirs:
                vid_feat_path = join(self.v_feat_dirs[0], vid_name + ".npz")
                if exists(vid_feat_path):
                    vid_feat = np.load(vid_feat_path)["features"]
                    self.v_feat_dim = vid_feat.shape[1]
                    
                    # Add dimensions for additional video features
                    for feat_dir in self.v_feat_dirs[1:]:
                        if feat_dir:
                            feat_path = join(feat_dir, vid_name + ".npz")
                            if exists(feat_path):
                                additional_feat = np.load(feat_path)["features"]
                                self.v_feat_dim += additional_feat.shape[1]
                    break
                    
        # Try to get audio feature dimension
        if self.a_feat_dir:
            for meta in self.data:
                vid_name = meta["vid"]
                # Try both .npy and .npz extensions
                aud_feat_path_npy = join(self.a_feat_dir, vid_name + ".npy")
                aud_feat_path_npz = join(self.a_feat_dir, vid_name + ".npz")
                
                if exists(aud_feat_path_npy):
                    aud_feat = np.load(aud_feat_path_npy)
                    self.a_feat_dim = aud_feat.shape[1] if len(aud_feat.shape) > 1 else aud_feat.shape[0]
                    break
                elif exists(aud_feat_path_npz):
                    aud_feat = np.load(aud_feat_path_npz)["features"]
                    self.a_feat_dim = aud_feat.shape[1]
                    break
                    
        if self.v_feat_dim is None:
            raise ValueError("Could not determine video feature dimension")
        if self.a_feat_dim is None:
            raise ValueError("Could not determine audio feature dimension")
            
        # Add dimension for temporal features if needed
        if self.use_tef:
            self.v_feat_dim += 2
            self.a_feat_dim += 2  # Add TEF to audio features as well for UMT

    def __getitem__(self, index):
        meta = self.data[index]
        model_inputs = dict()
        
        # Get query features
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])
        
        # Get video features (required for UMT)
        ctx_l = meta["duration"]
        vid_feat_path = join(self.v_feat_dirs[0], meta["vid"] + ".npz")
        if exists(vid_feat_path):
            vid_feat = np.load(vid_feat_path)["features"][:ctx_l]
            if self.normalize_v:
                vid_feat = l2_normalize_np_array(vid_feat)
        else:
            vid_feat = np.zeros((ctx_l, self.v_feat_dim))
            logger.warning(f"Video feature not found: {vid_feat_path}")
            
        # Concatenate multiple video features if available
        for i, feat_dir in enumerate(self.v_feat_dirs[1:], start=1):
            if feat_dir:
                feat_path = join(feat_dir, meta["vid"] + ".npz")
                if exists(feat_path):
                    additional_feat = np.load(feat_path)["features"][:ctx_l]
                    if self.normalize_v:
                        additional_feat = l2_normalize_np_array(additional_feat)
                    
                    # Ensure both features have the same length before concatenation
                    min_len = min(vid_feat.shape[0], additional_feat.shape[0])
                    vid_feat = vid_feat[:min_len]
                    additional_feat = additional_feat[:min_len]
                    
                    vid_feat = np.concatenate([vid_feat, additional_feat], axis=1)
                else:
                    logger.warning(f"Additional video feature not found: {feat_path}")
                    
        model_inputs["video_feat"] = vid_feat
        
        # Get audio features (required for UMT)
        aud_feat_path_npy = join(self.a_feat_dir, meta["vid"] + ".npy")
        aud_feat_path_npz = join(self.a_feat_dir, meta["vid"] + ".npz")
        
        if exists(aud_feat_path_npy):
            aud_feat = np.load(aud_feat_path_npy)[:ctx_l]
            if self.normalize_v:  # Use same normalization flag for audio
                aud_feat = l2_normalize_np_array(aud_feat)
        elif exists(aud_feat_path_npz):
            aud_feat = np.load(aud_feat_path_npz)["features"][:ctx_l]
            if self.normalize_v:  # Use same normalization flag for audio
                aud_feat = l2_normalize_np_array(aud_feat)
        else:
            aud_feat = np.zeros((ctx_l, self.a_feat_dim))
            logger.warning(f"Audio feature not found: {aud_feat_path_npy} or {aud_feat_path_npz}")
            
        model_inputs["audio_feat"] = aud_feat
        
        # Ensure both features have the same length
        min_len = min(vid_feat.shape[0], aud_feat.shape[0])
        model_inputs["video_feat"] = vid_feat[:min_len]
        model_inputs["audio_feat"] = aud_feat[:min_len]
        ctx_l = min_len
        
        # Add temporal features if needed
        if self.use_tef:
            tef_st = np.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = np.stack([tef_st, tef_ed], axis=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = np.concatenate(
                    [model_inputs["video_feat"], tef], axis=1)  # (Lv, Dv+2)
            # Add TEF to audio features as well for UMT
            model_inputs["audio_feat"] = np.concatenate(
                [model_inputs["audio_feat"], tef], axis=1)  # (La, Da+2)
        
        # Process labels if needed
        if self.load_labels:
            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)
        
        # Return in the expected format - separate meta from model_inputs
        return {
            "meta": meta,
            "model_inputs": model_inputs
        }

    def _get_query_feat_by_qid(self, qid):
        """Get query feature by query id"""
        if self.dset_name == 'tvsum':
            q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
        else:
            # QVhighlight dataset uses qid prefix
            q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
        
        q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)
        if self.q_feat_type == "last_hidden_state":
            q_feat = q_feat[:self.max_q_l]
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list of windows, each window is [st, ed] in seconds
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) == 0:
            return torch.zeros(0, 2)
        windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx format
        windows = span_xx_to_cxw(windows)  # normalized windows in cxw format
        return windows


def start_end_collate_umt(batch):
    """
    Collate function for UMT dataset batches
    """
    batch_meta = [e["meta"] for e in batch]

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        elif k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        elif k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue
        elif k == "query_feat":
            # query_feat is already a torch tensor
            batched_data[k] = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
            continue
        else:
            # For features (video_feat, audio_feat) - these are numpy arrays
            # Convert to torch tensors after padding
            padded_data, mask_data = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = (torch.from_numpy(padded_data), torch.from_numpy(mask_data))
    
    return batch_meta, batched_data


def prepare_batch_inputs_umt(batched_model_inputs, device, non_blocking=False):
    """
    Prepare batch inputs for UMT model
    """
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        src_aud=batched_model_inputs["audio_feat"][0].to(device, non_blocking=non_blocking),
        src_aud_mask=batched_model_inputs["audio_feat"][1].to(device, non_blocking=non_blocking),
    )
    
    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
