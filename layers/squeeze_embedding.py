import torch
import torch.nn as nn
import numpy as np


class SqueezeEmbedding(nn.Module):

  def __init__(self, batch_first=True):
    super(SqueezeEmbedding, self).__init__()
    self.batch_first = batch_first

  def forward(self, x, x_len):
    """sort"""
    x_sort_idx = torch.sort(-x_len)[1].long()
    x_unsort_idx = torch.sort(x_sort_idx)[1].long()
    x_len = x_len[x_sort_idx]
    x = x[x_sort_idx]
    """pack"""
    x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(
        x, x_len, batch_first=self.batch_first)
    """unpack: out"""
    out = torch.nn.utils.rnn.pad_packed_sequence(
        x_emb_p, batch_first=self.batch_first)  # (sequence, lengths)
    out = out[0]  #
    """unsort"""
    out = out[x_unsort_idx]
    return out
