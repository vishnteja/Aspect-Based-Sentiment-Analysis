import torch
from torch import nn
import torch.nn.functional as F
from layers.glu import GLU_Block
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention


class GC_IAN1(nn.Module):

  def __init__(self, embedding_matrix, opt):
    super(GC_IAN, self).__init__()
    self.opt = opt
    kernel_sizes = [1, 3, 5, 7]
    self.embed = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float))

    self.cnn_context = nn.ModuleList([
        self.cnn_layer(opt.embed_dim, opt.in_channels, k) for k in kernel_sizes
    ])

    self.cnn_context_ = nn.ModuleList([
        self.cnn_layer(opt.embed_dim, opt.in_channels, k) for k in kernel_sizes
    ])

    self.cnn_aspect = nn.ModuleList([
        self.cnn_layer(opt.embed_dim, opt.in_channels, k) for k in kernel_sizes
    ])

    self.attention_1 = Attention(
        len(kernel_sizes) * opt.in_channels, score_function='bi_linear')
    self.attention_2 = Attention(
        len(kernel_sizes) * opt.in_channels, score_function='bi_linear')

    self.dropout = nn.Dropout(opt.dropout)
    self.dense = nn.Linear(2 * len(kernel_sizes) * opt.in_channels,
                           opt.polarities_dim)

  def cnn_layer(self, in_channels, out_channels, k):
    layers = [
        nn.ConstantPad2d((0, 0, k - 1, 0), 0),
        nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1))
    ]
    return nn.Sequential(*layers)

  def forward(self, inputs):
    text_raw_indices = inputs[0]
    aspect_indices = inputs[1]

    context_len = torch.tensor(
        torch.sum(text_raw_indices != 0, dim=-1),
        dtype=torch.float).to(self.opt.device)
    aspect_len = torch.tensor(
        torch.sum(aspect_indices != 0, dim=-1),
        dtype=torch.float).to(self.opt.device)
    context = self.embed(text_raw_indices)  # [batch, max_seq_len, embed_dim]
    aspect = self.embed(aspect_indices)  # [batch, max_seq_len, embed_dim]

    # Aspect Vector
    aspect = torch.transpose(aspect, 1, 2)  # [batch, embed_dim, max_asp_len]
    aspect = aspect.unsqueeze(3)  # [batch, embed_dim, max_asp_len, 1]
    aspect = [conv(aspect) for conv in self.cnn_aspect]

    # Aspect Pool
    aspect_pool = [
        torch.max_pool1d(torch.squeeze(a, 3), kernel_size=a.shape[2])
        for a in aspect
    ]

    # Context Vector
    context = torch.transpose(context, 1, 2)
    context_v = context.unsqueeze(3)
    context = [conv(context_v) for conv in self.cnn_context]
    context_ = [conv(context_v) for conv in self.cnn_context_]

    # Context Pool
    context_pool = [
        torch.max_pool1d(torch.squeeze(c, 3), kernel_size=c.shape[2])
        for c in context
    ]

    # Gating
    s1 = [
        torch.tanh(c) * torch.sigmoid(a)
        for c, a, c_ in zip(context, aspect, context_)
    ]
    s2 = [
        torch.tanh(a) * torch.sigmoid(a)
        for c, a, c_ in zip(context, aspect, context_)
    ]
    s1 = torch.cat(s1, 1)
    s2 = torch.cat(s2, 1)

    res1 = torch.max_pool1d(s1.squeeze(3), kernel_size=s1.shape[2]).squeeze(2)
    res2 = torch.max_pool1d(s2.squeeze(3), kernel_size=s2.shape[2]).squeeze(2)

    x = torch.cat((res1, res2), dim=-1)
    x = self.dropout(x)
    out = self.dense(x)
    return out


class GC_IAN2(nn.Module):

  def __init__(self, embedding_matrix, opt):
    super(GC_IAN, self).__init__()
    self.opt = opt
    kernel_sizes = [1, 3, 5, 7]
    self.embed = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float))

    self.cnn_context = nn.ModuleList([
        self.cnn_layer(opt.embed_dim, opt.in_channels, k) for k in kernel_sizes
    ])

    self.cnn_context_ = nn.ModuleList([
        self.cnn_layer(opt.embed_dim, opt.in_channels, k) for k in kernel_sizes
    ])

    self.cnn_aspect = nn.ModuleList([
        self.cnn_layer(opt.embed_dim, opt.in_channels, k) for k in kernel_sizes
    ])

    self.attention_1 = Attention(
        len(kernel_sizes) * opt.in_channels, score_function='bi_linear')
    self.attention_2 = Attention(
        len(kernel_sizes) * opt.in_channels, score_function='bi_linear')

    self.dropout = nn.Dropout(opt.dropout)
    self.dense = nn.Linear(2 * len(kernel_sizes) * opt.in_channels,
                           opt.polarities_dim)

  def cnn_layer(self, in_channels, out_channels, k):
    layers = [
        nn.ConstantPad2d((0, 0, k - 1, 0), 0),
        nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1))
    ]
    return nn.Sequential(*layers)

  def forward(self, inputs):
    text_raw_indices = inputs[0]
    aspect_indices = inputs[1]

    context_len = torch.tensor(
        torch.sum(text_raw_indices != 0, dim=-1),
        dtype=torch.float).to(self.opt.device)
    aspect_len = torch.tensor(
        torch.sum(aspect_indices != 0, dim=-1),
        dtype=torch.float).to(self.opt.device)
    context = self.embed(text_raw_indices)  # [batch, max_seq_len, embed_dim]
    aspect = self.embed(aspect_indices)  # [batch, max_seq_len, embed_dim]

    # Aspect Vector
    aspect = torch.transpose(aspect, 1, 2)  # [batch, embed_dim, max_asp_len]
    aspect = aspect.unsqueeze(3)  # [batch, embed_dim, max_asp_len, 1]
    aspect = [conv(aspect) for conv in self.cnn_aspect]

    # Aspect Pool
    aspect_pool = [
        torch.max_pool1d(torch.squeeze(a, 3), kernel_size=a.shape[2])
        for a in aspect
    ]

    # Context Vector
    context = torch.transpose(context, 1, 2)
    context_v = context.unsqueeze(3)
    context = [conv(context_v) for conv in self.cnn_context]
    context_ = [conv(context_v) for conv in self.cnn_context_]

    # Context Pool
    context_pool = [
        torch.max_pool1d(torch.squeeze(c, 3), kernel_size=c.shape[2])
        for c in context
    ]

    # Gating
    s1 = [
        torch.tanh(c) * torch.sigmoid(a + c_)
        for c, a, c_ in zip(context, aspect, context_)
    ]
    s2 = [
        torch.tanh(a) * torch.sigmoid(a)
        for c, a, c_ in zip(context, aspect, context_)
    ]
    s1 = torch.cat(s1, 1)
    s2 = torch.cat(s2, 1)

    res1 = torch.max_pool1d(s1.squeeze(3), kernel_size=s1.shape[2]).squeeze(2)
    res2 = torch.max_pool1d(s2.squeeze(3), kernel_size=s2.shape[2]).squeeze(2)

    x = torch.cat((res1, res2), dim=-1)
    x = self.dropout(x)
    out = self.dense(x)
    return out


class GC_IAN3(nn.Module):

  def __init__(self, embedding_matrix, opt):
    super(GC_IAN, self).__init__()
    self.opt = opt
    kernel_sizes = [1, 3, 5, 7]
    self.embed = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float))

    self.cnn_context = nn.ModuleList([
        self.cnn_layer(opt.embed_dim, opt.in_channels, k) for k in kernel_sizes
    ])

    self.cnn_context_ = nn.ModuleList([
        self.cnn_layer(opt.embed_dim, opt.in_channels, k) for k in kernel_sizes
    ])

    self.cnn_aspect = nn.ModuleList([
        self.cnn_layer(opt.embed_dim, opt.in_channels, k) for k in kernel_sizes
    ])

    self.attention_1 = Attention(
        len(kernel_sizes) * opt.in_channels, score_function='bi_linear')
    self.attention_2 = Attention(
        len(kernel_sizes) * opt.in_channels, score_function='bi_linear')

    self.dropout = nn.Dropout(opt.dropout)
    self.dense = nn.Linear(2 * len(kernel_sizes) * opt.in_channels,
                           opt.polarities_dim)

  def cnn_layer(self, in_channels, out_channels, k):
    layers = [
        nn.ConstantPad2d((0, 0, k - 1, 0), 0),
        nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1))
    ]
    return nn.Sequential(*layers)

  def forward(self, inputs):
    text_raw_indices = inputs[0]
    aspect_indices = inputs[1]

    context_len = torch.tensor(
        torch.sum(text_raw_indices != 0, dim=-1),
        dtype=torch.float).to(self.opt.device)
    aspect_len = torch.tensor(
        torch.sum(aspect_indices != 0, dim=-1),
        dtype=torch.float).to(self.opt.device)
    context = self.embed(text_raw_indices)  # [batch, max_seq_len, embed_dim]
    aspect = self.embed(aspect_indices)  # [batch, max_seq_len, embed_dim]

    # Aspect Vector
    aspect = torch.transpose(aspect, 1, 2)  # [batch, embed_dim, max_asp_len]
    aspect = aspect.unsqueeze(3)  # [batch, embed_dim, max_asp_len, 1]
    aspect = [conv(aspect) for conv in self.cnn_aspect]

    # Aspect Pool
    aspect_pool = [
        torch.max_pool1d(torch.squeeze(a, 3), kernel_size=a.shape[2])
        for a in aspect
    ]

    # Context Vector
    context = torch.transpose(context, 1, 2)
    context_v = context.unsqueeze(3)
    context = [conv(context_v) for conv in self.cnn_context]
    context_ = [conv(context_v) for conv in self.cnn_context_]

    # Context Pool
    context_pool = [
        torch.max_pool1d(torch.squeeze(c, 3), kernel_size=c.shape[2])
        for c in context
    ]

    # Gating
    s1 = [
        torch.tanh(c) * torch.sigmoid(a + c_)
        for c, a, c_ in zip(context, aspect, context_)
    ]
    s2 = [
        torch.tanh(a) * torch.sigmoid(a + c)
        for c, a, c_ in zip(context, aspect, context_)
    ]
    s1 = torch.cat(s1, 1)
    s2 = torch.cat(s2, 1)

    res1 = torch.max_pool1d(s1.squeeze(3), kernel_size=s1.shape[2]).squeeze(2)
    res2 = torch.max_pool1d(s2.squeeze(3), kernel_size=s2.shape[2]).squeeze(2)

    x = torch.cat((res1, res2), dim=-1)
    x = self.dropout(x)
    out = self.dense(x)
    return out