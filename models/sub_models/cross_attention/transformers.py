"""Modified from DETR's transformer.py

- Cross encoder layer is similar to the decoder layers in Transformer, but
  updates both source and target features
- Added argument to control whether value has position embedding or not for
  TransformerEncoderLayer and TransformerDecoderLayer
- Decoder layer now keeps track of attention weights
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerCrossEncoder(nn.Module):

    def __init__(self, cross_encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(cross_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                tgt_xyz: Optional[Tensor] = None):

        src_intermediate, tgt_intermediate = [], []

        for layer in self.layers:
            src, tgt = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos, tgt_pos=tgt_pos)
            if self.return_intermediate:
                src_intermediate.append(self.norm(src) if self.norm is not None else src)
                tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        if self.norm is not None:
            src = self.norm(src)
            tgt = self.norm(tgt)
            if self.return_intermediate:
                if len(self.layers) > 0:
                    src_intermediate.pop()
                    tgt_intermediate.pop()
                src_intermediate.append(src)
                tgt_intermediate.append(tgt)

        if self.return_intermediate:
            return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        return src, tgt

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)

class MaskedTransformerCrossEncoder(TransformerCrossEncoder):
    
    def __init__(self, cross_encoder_layer, num_layers, masking_radius, norm=None, return_intermediate=False):
        super().__init__(cross_encoder_layer, num_layers)
        self.layers = _get_clones(cross_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert len(masking_radius) == num_layers
        self.masking_radius = masking_radius

    def compute_mask(self, xyz, radius):
        with torch.no_grad():
            dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist
    
    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                tgt_xyz: Optional[Tensor] = None):

        src_intermediate, tgt_intermediate = [], []

        for idx, layer in enumerate(self.layers):
            if self.masking_radius[idx] > 0:
                src_mask, src_dist = self.compute_mask(src_xyz, self.masking_radius[idx])
                tgt_mask, src_dist = self.compute_mask(tgt_xyz, self.masking_radius[idx])
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = src_mask.shape
                nhead = layer.nhead
                src_mask = src_mask.unsqueeze(1)
                src_mask = src_mask.repeat(1, nhead, 1, 1)
                src_mask = src_mask.view(bsz * nhead, n, n)
                tgt_mask = tgt_mask.unsqueeze(1)
                tgt_mask = tgt_mask.repeat(1, nhead, 1, 1)
                tgt_mask = tgt_mask.view(bsz * nhead, n, n)
                
            src, tgt = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos, tgt_pos=tgt_pos)
            if self.return_intermediate:
                src_intermediate.append(self.norm(src) if self.norm is not None else src)
                tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        if self.norm is not None:
            src = self.norm(src)
            tgt = self.norm(tgt)
            if self.return_intermediate:
                if len(self.layers) > 0:
                    src_intermediate.pop()
                    tgt_intermediate.pop()
                src_intermediate.append(src)
                tgt_intermediate.append(tgt)

        if self.return_intermediate:
            return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        return src, tgt

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)

class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        self.nhead = nhead

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):


        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                   key_padding_mask=src_key_padding_mask)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,):


        # Self attention
        src2 = self.norm1(src)
        src2_w_pos = self.with_pos_embed(src2, src_pos)
        q = k = src2_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                                              value=src2_w_pos if self.sa_val_has_pos_emb else src2,
                                              attn_mask=src_mask,
                                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        tgt2 = self.norm1(tgt)
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)
        q = k = tgt2_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        # Cross attention
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        src3, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src2, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt3, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src2,
                                                   key_padding_mask=src_key_padding_mask)

        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)

        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)

class FlexibleTransformerEncoder(nn.Module):
    
    def __init__(self, self_layer, cross_layer, layer_list, masking_radius, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _flex_clones(self_layer, cross_layer, layer_list)
        self.num_layers = len(layer_list)
        self.layer_list = layer_list
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert len(masking_radius) == self.layer_list.count('s')
        self.masking_radius = masking_radius

    def compute_mask(self, xyz, radius):
        with torch.no_grad():
            dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist
    
    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_xyz: Optional[Tensor] = None, 
                tgt_xyz: Optional[Tensor] = None):

        src_intermediate, tgt_intermediate = [], []
        
        s_count=0

        for idx, layer in enumerate(self.layers):
            if self.layer_list[idx]=='s' and self.masking_radius[s_count] > 0:
                src_mask, src_dist = self.compute_mask(src_xyz, self.masking_radius[s_count])
                tgt_mask, src_dist = self.compute_mask(tgt_xyz, self.masking_radius[s_count])
                s_count += 1
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = src_mask.shape
                nhead = layer.nhead
                src_mask = src_mask.unsqueeze(1)
                src_mask = src_mask.repeat(1, nhead, 1, 1)
                src_mask = src_mask.view(bsz * nhead, n, n)
                tgt_mask = tgt_mask.unsqueeze(1)
                tgt_mask = tgt_mask.repeat(1, nhead, 1, 1)
                tgt_mask = tgt_mask.view(bsz * nhead, n, n)
                
            src, tgt = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos, tgt_pos=tgt_pos)
            if self.return_intermediate:
                src_intermediate.append(self.norm(src) if self.norm is not None else src)
                tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        if self.norm is not None:
            src = self.norm(src)
            tgt = self.norm(tgt)
            if self.return_intermediate:
                if len(self.layers) > 0:
                    src_intermediate.pop()
                    tgt_intermediate.pop()
                src_intermediate.append(src)
                tgt_intermediate.append(tgt)

        if self.return_intermediate:
            return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        return src, tgt

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)

class TransformerSelfLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        self.nhead = nhead

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):


        # Self attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                              value=src_w_pos if self.sa_val_has_pos_emb else src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,):


        # Self attention
        src2 = self.norm1(src)
        src2_w_pos = self.with_pos_embed(src2, src_pos)
        q = k = src2_w_pos
        src2, satt_weights_s = self.self_attn(q, k,
                                              value=src2_w_pos if self.sa_val_has_pos_emb else src2,
                                              attn_mask=src_mask,
                                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        tgt2 = self.norm1(tgt)
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)
        q = k = tgt2_w_pos
        tgt2, satt_weights_t = self.self_attn(q, k,
                                              value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
                                              attn_mask=tgt_mask,
                                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)


        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        
class TransformerCrossLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        self.nhead = nhead

        # Self, cross attention layers
        if attention_type == 'dot_prod':
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,):

        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                   key_padding_mask=src_key_padding_mask)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # Stores the attention weights for analysis, if required
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,):

        # Cross attention
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        src3, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src2, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt3, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src2,
                                                   key_padding_mask=src_key_padding_mask)

        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)

        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask, src_pos, tgt_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _flex_clones(self_layer, cross_layer, layer_list):
    module_list = nn.ModuleList([])
    for i in layer_list:
        if i == "s":
            module_list.append(copy.deepcopy(self_layer))
        elif i == "c":
            module_list.append(copy.deepcopy(cross_layer))
        else:
            assert(i in ["s", "c"]), "Please set layer_list only with 's' and 'c' representing 'self_attention_layer' and 'cross_attention_layer' respectively"
    return module_list


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

