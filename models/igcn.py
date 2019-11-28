import torch
from torch import nn
import torch.nn.functional as F
from layers.igcn_utils import *


class IGCN(nn.Module):

  def __init__(self, embedding_matrix, pos_matrix, opt):
    super(IGCN, self).__init__()
    self.opt = opt

    # Load Pretrained Embeddings
    opt.max_seq_len += 100
    self.embed = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float))
    self.pos_embed = nn.Embedding.from_pretrained(
        torch.tensor(pos_matrix, dtype=torch.float))
    self.position_embed = nn.Embedding(opt.max_seq_len, opt.position_dim)
    opt.embed_dim += 36
    opt.position_dim = 100

    self.convres_aspect = nn.utils.weight_norm(
        nn.Conv2d(opt.embed_dim, opt.in_channels, kernel_size=(1, 1)),
        name='weight',
        dim=0)
    self.convres_context = nn.utils.weight_norm(
        nn.Conv2d(
            opt.embed_dim + opt.position_dim,
            opt.in_channels,
            kernel_size=(1, 1)),
        name='weight',
        dim=0)

    # Context CNN
    self.cnn_context = nn.ModuleList([
        cnn_layer(opt.embed_dim + opt.position_dim, opt.in_channels, k)
        for k in kernel_sizes_context
    ])
    self.cnn_context2 = nn.ModuleList([
        cnn_layer(opt.embed_dim + opt.position_dim, opt.in_channels, k)
        for k in kernel_sizes_context
    ])
    self.fc_context_gate = nn.Linear(opt.in_channels, opt.in_channels)

    # Aspect CNN
    self.cnn_aspect = nn.ModuleList([
        cnn_layer(opt.embed_dim, opt.in_channels, k)
        for k in kernel_sizes_aspect
    ])
    self.cnn_aspect2 = nn.ModuleList([
        cnn_layer(opt.embed_dim, opt.in_channels, k)
        for k in kernel_sizes_aspect
    ])
    self.fc_aspect_gate = nn.Linear(opt.in_channels, opt.in_channels)

    # Linear Layer
    final_dim = 2 * len(kernel_sizes_context) * opt.in_channels
    self.dropout = nn.Dropout(opt.dropout)
    self.dense = nn.Linear(final_dim, opt.polarities_dim)

  def forward(self, inputs):
    text_raw_indices = inputs[0]
    aspect_indices = inputs[1]
    pos_indices = inputs[2]
    aspect_pos_indices = inputs[3]
    position_indices = inputs[4]

    # Inputs
    context = self.embed(
        text_raw_indices)  # Dimensions: [batch, max_seq_len, embed_dim]
    aspect = self.embed(
        aspect_indices)  # Dimensions: [batch, max_seq_len, embed_dim]
    position = self.position_embed(position_indices)

    # Part-of-speech(POS) tags
    pos_tags = self.pos_embed(pos_indices)
    aspect_pos_tags = self.pos_embed(aspect_pos_indices)

    # Concat POS Tags
    context = torch.cat((context, pos_tags, position), dim=-1)
    aspect = torch.cat((aspect, aspect_pos_tags), dim=-1)

    # Aspect
    aspect = torch.transpose(aspect, 1, 2)  # [batch, embed_dim, max_asp_len]
    aspect_v = aspect.unsqueeze(3)  # [batch, embed_dim, max_asp_len, 1]
    aspect = [conv(aspect_v) for conv in self.cnn_aspect]
    aspect2 = [conv(aspect_v) for conv in self.cnn_aspect2]

    # Aspect Pool
    aspect_pool = [
        torch.max_pool1d(torch.squeeze(a, 3), kernel_size=a.shape[2])
        for a in aspect
    ]

    # Context
    context = torch.transpose(context, 1, 2)
    context_v = context.unsqueeze(3)
    context = [conv(context_v) for conv in self.cnn_context]
    context2 = [conv(context_v) for conv in self.cnn_context2]

    # Context Pool
    context_pool = [
        torch.max_pool1d(torch.squeeze(c, 3), kernel_size=c.shape[2])
        for c in context
    ]

    ##  Gating
    # Context Gate
    context_gate = [
        self.fc_context_gate(ap.squeeze(2)).unsqueeze_(2).unsqueeze_(3) + c2
        for ap, c2 in zip(aspect_pool, context2)
    ]
    aspect2context = gate(context, context_gate, option=self.opt.gating)

    # Aspect Gate
    aspect_gate = [
        self.fc_aspect_gate(cp.squeeze(2)).unsqueeze_(2).unsqueeze_(3) + a2
        for cp, a2 in zip(context_pool, aspect2)
    ]
    context2aspect = gate(aspect, aspect_gate, option=self.opt.gating)

    # Concatenate all kernel outputs
    aspect2context = torch.cat(aspect2context, 1)
    context2aspect = torch.cat(context2aspect, 1)

    # Maxpool
    final_context = torch.max_pool1d(
        aspect2context.squeeze(3),
        kernel_size=aspect2context.shape[2]).squeeze(2)
    final_aspect = torch.max_pool1d(
        context2aspect.squeeze(3),
        kernel_size=context2aspect.shape[2]).squeeze(2)

    # FeatureVec
    final = torch.cat((final_context, final_aspect), dim=-1)
    final = self.dropout(final)
    out = self.dense(final)
    return out


class IGCN_2(nn.Module):

  def __init__(self, embedding_matrix, pos_matrix, opt):
    super(IGCN_2, self).__init__()
    self.opt = opt

    # Load Pretrained Embeddings
    opt.max_seq_len += 100
    self.embed = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float))
    self.pos_embed = nn.Embedding.from_pretrained(
        torch.tensor(pos_matrix, dtype=torch.float))
    self.position_embed = nn.Embedding(opt.max_seq_len, opt.position_dim)
    opt.embed_dim += 36
    opt.position_dim = 100

    self.convres_aspect = nn.utils.weight_norm(
        nn.Conv2d(opt.embed_dim, opt.in_channels, kernel_size=(1, 1)),
        name='weight',
        dim=0)
    self.convres_context = nn.utils.weight_norm(
        nn.Conv2d(
            opt.embed_dim + opt.position_dim,
            opt.in_channels,
            kernel_size=(1, 1)),
        name='weight',
        dim=0)

    # Context CNN
    self.cnn_context = nn.ModuleList([
        cnn_layer(opt.embed_dim + opt.position_dim, opt.in_channels, k)
        for k in kernel_sizes_context
    ])
    self.cnn_context2 = nn.ModuleList([
        cnn_layer(opt.embed_dim + opt.position_dim, opt.in_channels, k)
        for k in kernel_sizes_context
    ])
    self.fc_context_gate = nn.Linear(opt.in_channels, opt.in_channels)

    # Aspect CNN
    self.cnn_aspect = nn.ModuleList([
        cnn_layer(opt.embed_dim, opt.in_channels, k)
        for k in kernel_sizes_aspect
    ])
    self.cnn_aspect2 = nn.ModuleList([
        cnn_layer(opt.embed_dim, opt.in_channels, k)
        for k in kernel_sizes_aspect
    ])
    self.fc_aspect_gate = nn.Linear(opt.in_channels, opt.in_channels)

    # Linear Layer
    final_dim = len(kernel_sizes_context) * opt.in_channels
    self.dropout = nn.Dropout(opt.dropout)
    self.dense = nn.Linear(final_dim, opt.polarities_dim)

  def forward(self, inputs):
    text_raw_indices = inputs[0]
    aspect_indices = inputs[1]
    pos_indices = inputs[2]
    aspect_pos_indices = inputs[3]
    position_indices = inputs[4]

    # Inputs
    context = self.embed(
        text_raw_indices)  # Dimensions: [batch, max_seq_len, embed_dim]
    aspect = self.embed(
        aspect_indices)  # Dimensions: [batch, max_seq_len, embed_dim]
    position = self.position_embed(position_indices)

    # Part-of-speech(POS) tags
    pos_tags = self.pos_embed(pos_indices)
    aspect_pos_tags = self.pos_embed(aspect_pos_indices)

    # Concat POS Tags
    context = torch.cat((context, pos_tags, position), dim=-1)
    aspect = torch.cat((aspect, aspect_pos_tags), dim=-1)

    # Aspect
    aspect = torch.transpose(aspect, 1, 2)  # [batch, embed_dim, max_asp_len]
    aspect_v = aspect.unsqueeze(3)  # [batch, embed_dim, max_asp_len, 1]
    aspect = [conv(aspect_v) for conv in self.cnn_aspect]
    aspect2 = [conv(aspect_v) for conv in self.cnn_aspect2]

    # Aspect Pool
    aspect_pool = [
        torch.max_pool1d(torch.squeeze(a, 3), kernel_size=a.shape[2])
        for a in aspect
    ]

    # Context
    context = torch.transpose(context, 1, 2)
    context_v = context.unsqueeze(3)
    context = [conv(context_v) for conv in self.cnn_context]
    context2 = [conv(context_v) for conv in self.cnn_context2]

    # Context Pool
    context_pool = [
        torch.max_pool1d(torch.squeeze(c, 3), kernel_size=c.shape[2])
        for c in context
    ]

    ##  Gating
    # Context Gate
    context_gate = [
        self.fc_context_gate(ap.squeeze(2)).unsqueeze_(2).unsqueeze_(3) + c2
        for ap, c2 in zip(aspect_pool, context2)
    ]
    aspect2context = gate(context, context_gate, option=self.opt.gating)

    # Aspect Gate
    # aspect_gate = [
    #     self.fc_aspect_gate(cp.squeeze(2)).unsqueeze_(2).unsqueeze_(3) + a2
    #     for cp, a2 in zip(context_pool, aspect2)
    # ]
    # context2aspect = gate(aspect, aspect_gate, option=self.opt.gating)

    # Concatenate all kernel outputs
    aspect2context = torch.cat(aspect2context, 1)
    # context2aspect = torch.cat(context2aspect, 1)

    # Maxpool
    final_context = torch.max_pool1d(
        aspect2context.squeeze(3),
        kernel_size=aspect2context.shape[2]).squeeze(2)
    # final_aspect = torch.max_pool1d(
    #     context2aspect.squeeze(3),
    #     kernel_size=context2aspect.shape[2]).squeeze(2)

    # FeatureVec
    final = torch.cat((final_context,), dim=-1)
    final = self.dropout(final)
    out = self.dense(final)
    return out


class IGCN_3(nn.Module):

  def __init__(self, embedding_matrix, pos_matrix, opt):
    super(IGCN_3, self).__init__()
    self.opt = opt

    # Load Pretrained Embeddings
    opt.max_seq_len += 100
    self.embed = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float))
    self.pos_embed = nn.Embedding.from_pretrained(
        torch.tensor(pos_matrix, dtype=torch.float))
    self.position_embed = nn.Embedding(opt.max_seq_len, opt.position_dim)
    opt.embed_dim += 36
    opt.position_dim = 100

    self.convres_aspect = nn.utils.weight_norm(
        nn.Conv2d(opt.embed_dim, opt.in_channels, kernel_size=(1, 1)),
        name='weight',
        dim=0)
    self.convres_context = nn.utils.weight_norm(
        nn.Conv2d(
            opt.embed_dim + opt.position_dim,
            opt.in_channels,
            kernel_size=(1, 1)),
        name='weight',
        dim=0)

    # Context CNN
    self.cnn_context = nn.ModuleList([
        cnn_layer(opt.embed_dim + opt.position_dim, opt.in_channels, k)
        for k in kernel_sizes_context
    ])
    self.cnn_context2 = nn.ModuleList([
        cnn_layer(opt.embed_dim + opt.position_dim, opt.in_channels, k)
        for k in kernel_sizes_context
    ])
    self.fc_context_gate = nn.Linear(opt.in_channels, opt.in_channels)

    # Aspect CNN
    self.cnn_aspect = nn.ModuleList([
        cnn_layer(opt.embed_dim, opt.in_channels, k)
        for k in kernel_sizes_aspect
    ])
    self.cnn_aspect2 = nn.ModuleList([
        cnn_layer(opt.embed_dim, opt.in_channels, k)
        for k in kernel_sizes_aspect
    ])
    self.fc_aspect_gate = nn.Linear(opt.in_channels, opt.in_channels)

    # Linear Layer
    final_dim = len(kernel_sizes_context) * opt.in_channels
    self.dropout = nn.Dropout(opt.dropout)
    self.dense = nn.Linear(final_dim, opt.polarities_dim)

  def forward(self, inputs):
    text_raw_indices = inputs[0]
    aspect_indices = inputs[1]
    pos_indices = inputs[2]
    aspect_pos_indices = inputs[3]
    position_indices = inputs[4]

    # Inputs
    context = self.embed(
        text_raw_indices)  # Dimensions: [batch, max_seq_len, embed_dim]
    aspect = self.embed(
        aspect_indices)  # Dimensions: [batch, max_seq_len, embed_dim]
    position = self.position_embed(position_indices)

    # Part-of-speech(POS) tags
    pos_tags = self.pos_embed(pos_indices)
    aspect_pos_tags = self.pos_embed(aspect_pos_indices)

    # Concat POS Tags
    context = torch.cat((context, pos_tags, position), dim=-1)
    aspect = torch.cat((aspect, aspect_pos_tags), dim=-1)

    # Aspect
    aspect = torch.transpose(aspect, 1, 2)  # [batch, embed_dim, max_asp_len]
    aspect_v = aspect.unsqueeze(3)  # [batch, embed_dim, max_asp_len, 1]
    aspect = [conv(aspect_v) for conv in self.cnn_aspect]
    aspect2 = [conv(aspect_v) for conv in self.cnn_aspect2]

    # Aspect Pool
    aspect_pool = [
        torch.max_pool1d(torch.squeeze(a, 3), kernel_size=a.shape[2])
        for a in aspect
    ]

    # Context
    context = torch.transpose(context, 1, 2)
    context_v = context.unsqueeze(3)
    context = [conv(context_v) for conv in self.cnn_context]
    context2 = [conv(context_v) for conv in self.cnn_context2]

    # Context Pool
    context_pool = [
        torch.max_pool1d(torch.squeeze(c, 3), kernel_size=c.shape[2])
        for c in context
    ]

    ##  Gating
    # Context Gate
    # context_gate = [
    #     self.fc_context_gate(ap.squeeze(2)).unsqueeze_(2).unsqueeze_(3) + c2
    #     for ap, c2 in zip(aspect_pool, context2)
    # ]
    # aspect2context = gate(context, context_gate, option=self.opt.gating)

    # Aspect Gate
    aspect_gate = [
        self.fc_aspect_gate(cp.squeeze(2)).unsqueeze_(2).unsqueeze_(3) + a2
        for cp, a2 in zip(context_pool, aspect2)
    ]
    context2aspect = gate(aspect, aspect_gate, option=self.opt.gating)

    # Concatenate all kernel outputs
    # aspect2context = torch.cat(aspect2context, 1)
    context2aspect = torch.cat(context2aspect, 1)

    # Maxpool
    # final_context = torch.max_pool1d(
    #     aspect2context.squeeze(3),
    #     kernel_size=aspect2context.shape[2]).squeeze(2)
    final_aspect = torch.max_pool1d(
        context2aspect.squeeze(3),
        kernel_size=context2aspect.shape[2]).squeeze(2)

    # FeatureVec
    final = torch.cat((final_aspect,), dim=-1)
    final = self.dropout(final)
    out = self.dense(final)
    return out


class IGCN_4(nn.Module):

  def __init__(self, embedding_matrix, pos_matrix, opt):
    super(IGCN_4, self).__init__()
    self.opt = opt

    # Load Pretrained Embeddings
    opt.max_seq_len += 100
    self.embed = nn.Embedding.from_pretrained(
        torch.tensor(embedding_matrix, dtype=torch.float))
    self.pos_embed = nn.Embedding.from_pretrained(
        torch.tensor(pos_matrix, dtype=torch.float))
    self.position_embed = nn.Embedding(opt.max_seq_len, opt.position_dim)
    opt.embed_dim += 36
    opt.position_dim = 100

    self.convres_aspect = nn.utils.weight_norm(
        nn.Conv2d(opt.embed_dim, opt.in_channels, kernel_size=(1, 1)),
        name='weight',
        dim=0)
    self.convres_context = nn.utils.weight_norm(
        nn.Conv2d(
            opt.embed_dim + opt.position_dim,
            opt.in_channels,
            kernel_size=(1, 1)),
        name='weight',
        dim=0)

    # Context CNN
    self.cnn_context = nn.ModuleList([
        cnn_layer(opt.embed_dim + opt.position_dim, opt.in_channels, k)
        for k in kernel_sizes_context
    ])
    self.cnn_context2 = nn.ModuleList([
        cnn_layer(opt.embed_dim + opt.position_dim, opt.in_channels, k)
        for k in kernel_sizes_context
    ])
    self.fc_context_gate = nn.Linear(opt.in_channels, opt.in_channels)

    # Aspect CNN
    self.cnn_aspect = nn.ModuleList([
        cnn_layer(opt.embed_dim, opt.in_channels, k)
        for k in kernel_sizes_aspect
    ])
    self.cnn_aspect2 = nn.ModuleList([
        cnn_layer(opt.embed_dim, opt.in_channels, k)
        for k in kernel_sizes_aspect
    ])
    self.fc_aspect_gate = nn.Linear(opt.in_channels, opt.in_channels)

    # Linear Layer
    final_dim = len(kernel_sizes_context) * opt.in_channels
    self.dropout = nn.Dropout(opt.dropout)
    self.dense = nn.Linear(final_dim, opt.polarities_dim)

  def forward(self, inputs):
    text_raw_indices = inputs[0]
    aspect_indices = inputs[1]
    pos_indices = inputs[2]
    aspect_pos_indices = inputs[3]
    position_indices = inputs[4]

    # Inputs
    context = self.embed(
        text_raw_indices)  # Dimensions: [batch, max_seq_len, embed_dim]
    aspect = self.embed(
        aspect_indices)  # Dimensions: [batch, max_seq_len, embed_dim]
    position = self.position_embed(position_indices)

    # Part-of-speech(POS) tags
    pos_tags = self.pos_embed(pos_indices)
    aspect_pos_tags = self.pos_embed(aspect_pos_indices)

    # Concat POS Tags
    context = torch.cat((context, pos_tags, position), dim=-1)
    aspect = torch.cat((aspect, aspect_pos_tags), dim=-1)

    # Aspect
    aspect = torch.transpose(aspect, 1, 2)  # [batch, embed_dim, max_asp_len]
    aspect_v = aspect.unsqueeze(3)  # [batch, embed_dim, max_asp_len, 1]
    aspect = [conv(aspect_v) for conv in self.cnn_aspect]
    aspect2 = [conv(aspect_v) for conv in self.cnn_aspect2]

    # Aspect Pool
    aspect_pool = [
        torch.max_pool1d(torch.squeeze(a, 3), kernel_size=a.shape[2])
        for a in aspect
    ]

    # Context
    context = torch.transpose(context, 1, 2)
    context_v = context.unsqueeze(3)
    context = [conv(context_v) for conv in self.cnn_context]
    context2 = [conv(context_v) for conv in self.cnn_context2]

    # Context Pool
    context_pool = [
        torch.max_pool1d(torch.squeeze(c, 3), kernel_size=c.shape[2])
        for c in context
    ]

    ##  Gating
    # Context Gate
    # context_gate = [
    #     self.fc_context_gate(ap.squeeze(2)).unsqueeze_(2).unsqueeze_(3) + c2
    #     for ap, c2 in zip(aspect_pool, context2)
    # ]
    # aspect2context = gate(context, context_gate, option=self.opt.gating)

    # Aspect Gate
    # aspect_gate = [
    #     self.fc_aspect_gate(cp.squeeze(2)).unsqueeze_(2).unsqueeze_(3) + a2
    #     for cp, a2 in zip(context_pool, aspect2)
    # ]
    # context2aspect = gate(aspect, aspect_gate, option=self.opt.gating)

    # Concatenate all kernel outputs
    aspect2context = torch.cat(context, 1)
    context2aspect = torch.cat(aspect, 1)

    # Maxpool
    final_context = torch.max_pool1d(
        aspect2context.squeeze(3),
        kernel_size=aspect2context.shape[2]).squeeze(2)
    final_aspect = torch.max_pool1d(
        context2aspect.squeeze(3),
        kernel_size=context2aspect.shape[2]).squeeze(2)

    # FeatureVec
    final = torch.cat((final_aspect,), dim=-1)
    final = self.dropout(final)
    out = self.dense(final)
    return out