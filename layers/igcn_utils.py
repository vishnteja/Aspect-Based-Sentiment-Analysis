import torch
import torch.nn as nn

kernel_sizes_context = [1, 3, 5]
kernel_sizes_aspect = [1, 1, 1]


def gate(context, aspect, option='GTU'):
  if option == 'GTU':
    gate_scores = [torch.sigmoid(a) for a in aspect]
    gate_scores = [torch.sum(g, dim=1) for g in gate_scores]
    gate_scores = [torch.softmax(g, dim=1) for g in gate_scores]
    res = [torch.tanh(c) * torch.sigmoid(a) for c, a in zip(context, aspect)]
  if option == 'GLU':
    gate_scores = [torch.sigmoid(a) for a in aspect]
    gate_scores = [torch.sum(g, dim=1) for g in gate_scores]
    gate_scores = [torch.softmax(g, dim=1) for g in gate_scores]
    res = [c * torch.sigmoid(a) for c, a, in zip(context, aspect)]
  if option == 'GTRU':
    gate_scores = [torch.relu(a) for a in aspect]
    gate_scores = [torch.sum(g, dim=1) for g in gate_scores]
    gate_scores = [torch.softmax(g, dim=1) for g in gate_scores]
    res = [torch.tanh(c) * torch.relu(a) for c, a, in zip(context, aspect)]

  gate_scores = [torch.transpose(g, 1, 2).squeeze_(1) for g in gate_scores]
  return res, gate_scores


def cnn_layer(in_channels, out_channels, k):
  layers = [
      nn.ConstantPad2d((0, 0, k - 1, 0), 0),
      nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1)),
      nn.LeakyReLU(),
      nn.Dropout(0.35),
      nn.ConstantPad2d((0, 0, k - 1, 0), 0),
      nn.Conv2d(out_channels, out_channels, kernel_size=(k, 1)),
      nn.LeakyReLU(),
      nn.Dropout(0.35)
  ]
  return nn.Sequential(*layers)
