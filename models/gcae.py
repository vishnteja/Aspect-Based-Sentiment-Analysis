import torch
from torch import nn
import torch.nn.functional as F
from layers.glu import GLU_Block
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention


class GCAE(nn.Module):

  def __init__(self, embedding_matrix, opt):
    super(GCAE, self).__init__()
    self.embed = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float))
    self.squeeze_embedding = SqueezeEmbedding()
    self.cnn_context_x = nn.ModuleList(
        [self.cnn_layer(opt, k) for k in [3, 4, 5]])
    self.cnn_context_y = nn.ModuleList(
        [self.cnn_layer(opt, k) for k in [3, 4, 5]])
    self.cnn_aspect = nn.Sequential(
        nn.ConstantPad2d((0, 0, 2, 0), 0),
        nn.Conv2d(opt.embed_dim, opt.in_channels, (3, 1)))
    self.dropout = nn.Dropout(opt.dropout)
    self.dense = nn.Linear(3 * opt.in_channels, opt.polarities_dim)

  def cnn_layer(self, opt, k):
    layers = [
        nn.ConstantPad2d((0, 0, k - 1, 0), 0),
        nn.Conv2d(opt.embed_dim, opt.in_channels, kernel_size=(k, 1))
    ]
    return nn.Sequential(*layers)

  def forward(self, inputs):
    text_raw_indices = inputs[0]
    aspect_indices = inputs[1]

    context_len = torch.sum(text_raw_indices != 0, dim=-1)
    aspect_len = torch.sum(aspect_indices != 0, dim=-1)
    context = self.embed(text_raw_indices)  # [batch, max_seq_len, embed_dim]
    aspect = self.embed(aspect_indices)  # [batch, max_seq_len, embed_dim]

    # Aspect Vector max pool over time
    aspect = torch.transpose(aspect, 1, 2)  # [batch, embed_dim, max_asp_len]
    aspect = aspect.unsqueeze(3)  # [batch, embed_dim, max_asp_len, 1]
    aspect = F.relu(
        self.cnn_aspect(aspect))  # [batch, in_channels, max_asp_len, 1]
    aspect_v = torch.max_pool1d(
        torch.squeeze(aspect, 3), kernel_size=aspect.shape[2])
    aspect_v = aspect_v.unsqueeze(2)

    # Context Vector
    context = torch.transpose(context, 1, 2)  # [batch, embed_dim, max_seq_len]
    context = context.unsqueeze(3)  # [batch, embed_dim, max_seq_len ,1]
    aspect_v = aspect_v.expand(-1, -1, context.shape[2], -1)
    x = [torch.tanh(conv(context)) for conv in self.cnn_context_x]
    y = [F.relu(conv(context) + aspect_v) for conv in self.cnn_context_y]
    res = [i * j for i, j in zip(x, y)]
    res = torch.cat(res, 1)
    res = torch.max_pool1d(torch.squeeze(res, 3), kernel_size=res.shape[2])
    res = res.squeeze(2)
    res = self.dropout(res)
    out = self.dense(res)
    return out
