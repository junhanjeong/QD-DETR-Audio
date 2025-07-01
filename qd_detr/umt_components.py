"""
UMT (Unified Multi-modal Transformers) components for QD-DETR-Audio
Based on UMT paper: https://arxiv.org/abs/2203.12745
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attn_output)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class BottleneckTransformerLayer(nn.Module):
    """
    Bottleneck Transformer Layer from UMT
    """
    def __init__(self, d_model, n_heads=8, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head attention layers
        self.att1 = MultiHeadAttention(d_model, n_heads, dropout)  # audio->token
        self.att2 = MultiHeadAttention(d_model, n_heads, dropout)  # video->token  
        self.att3 = MultiHeadAttention(d_model, n_heads, dropout)  # token->audio
        self.att4 = MultiHeadAttention(d_model, n_heads, dropout)  # token->video
        
        # Feed forward networks
        self.ffn1 = FeedForwardNetwork(d_model, d_ff, dropout)  # for audio
        self.ffn2 = FeedForwardNetwork(d_model, d_ff, dropout)  # for video
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)  # audio
        self.norm2 = nn.LayerNorm(d_model)  # video
        self.norm3 = nn.LayerNorm(d_model)  # token
        self.norm4 = nn.LayerNorm(d_model)  # token after cross-attention
        self.norm5 = nn.LayerNorm(d_model)  # audio before ffn
        self.norm6 = nn.LayerNorm(d_model)  # video before ffn
        
    def forward(self, audio, video, token, pos_embed=None, mask=None):
        """
        Args:
            audio: [batch_size, audio_len, d_model]
            video: [batch_size, video_len, d_model] 
            token: [batch_size, num_tokens, d_model]
            pos_embed: positional embedding
            mask: attention mask
        """
        # Normalize inputs
        da = self.norm1(audio)
        dv = self.norm2(video)
        dt = self.norm3(token)
        
        # Add positional embeddings if provided
        ka = da + pos_embed if pos_embed is not None else da
        kv = dv + pos_embed if pos_embed is not None else dv
        
        # Cross attention: token attends to audio and video
        at = self.att1(dt, ka, da, mask)  # token <- audio
        vt = self.att2(dt, kv, dv, mask)  # token <- video
        
        # Update token
        token = token + at + vt
        dt = self.norm4(token)
        
        # Self attention: audio and video attend to updated token
        qa = da + pos_embed if pos_embed is not None else da
        qv = dv + pos_embed if pos_embed is not None else dv
        
        audio = audio + self.att3(qa, dt, dt)  # audio <- token
        video = video + self.att4(qv, dt, dt)  # video <- token
        
        # Feed forward
        da = self.norm5(audio)
        dv = self.norm6(video)
        
        audio = audio + self.ffn1(da)
        video = video + self.ffn2(dv)
        
        return audio, video, token


class BottleneckTransformer(nn.Module):
    """
    Bottleneck Transformer from UMT
    """
    def __init__(self, d_model, num_tokens=4, num_layers=1, n_heads=8, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        
        # Learnable bottleneck tokens
        self.token = nn.Parameter(torch.randn(num_tokens, d_model))
        nn.init.xavier_uniform_(self.token)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            BottleneckTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, audio, video, pos_embed=None, mask=None):
        """
        Args:
            audio: [batch_size, audio_len, d_model]
            video: [batch_size, video_len, d_model]
            pos_embed: positional embedding
            mask: attention mask
        """
        batch_size = audio.size(0)
        
        # Expand tokens for batch
        token = self.token.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply bottleneck transformer layers
        for layer in self.layers:
            audio, video, token = layer(audio, video, token, pos_embed, mask)
            
        return audio, video


class UniModalEncoder(nn.Module):
    """
    Unimodal encoder from UMT
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len]
        """
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Apply transformer encoder
        if mask is not None:
            # Convert mask to attention mask (inverted)
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
            
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Apply final norm
        x = self.norm(x)
        
        return x


class CrossModalEncoder(nn.Module):
    """
    Cross-modal encoder from UMT
    """
    def __init__(self, hidden_dim, num_tokens=4, num_layers=1, n_heads=8, dropout=0.1, fusion_type='sum'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        
        # Bottleneck transformer
        self.bottleneck = BottleneckTransformer(
            d_model=hidden_dim,
            num_tokens=num_tokens,
            num_layers=num_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Fusion layer for concatenation
        if fusion_type == 'concat':
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Final norm
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, audio, video, mask=None):
        """
        Args:
            audio: [batch_size, audio_len, hidden_dim]
            video: [batch_size, video_len, hidden_dim]
            mask: attention mask
        """
        # Apply bottleneck transformer
        audio_refined, video_refined = self.bottleneck(audio, video, mask=mask)
        
        # Fusion
        if self.fusion_type == 'sum':
            # Average pooling then sum
            audio_pooled = audio_refined.mean(dim=1)  # [batch_size, hidden_dim]
            video_pooled = video_refined.mean(dim=1)  # [batch_size, hidden_dim]
            fused = audio_pooled + video_pooled
        elif self.fusion_type == 'mean':
            audio_pooled = audio_refined.mean(dim=1)
            video_pooled = video_refined.mean(dim=1)
            fused = (audio_pooled + video_pooled) / 2
        elif self.fusion_type == 'concat':
            audio_pooled = audio_refined.mean(dim=1)
            video_pooled = video_refined.mean(dim=1)
            fused = torch.cat([audio_pooled, video_pooled], dim=-1)
            fused = self.fusion_proj(fused)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
            
        # Apply final norm
        fused = self.norm(fused)
        
        return fused
