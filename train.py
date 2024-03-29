import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy
import config

# from pytorch_transformers import BertModel
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, build_pos_tagger, ABSADataset

from models import LSTM, IAN, TD_LSTM, TC_LSTM, AT_LSTM, ATAE_LSTM, AOA, Gated_CNN, GCAE
from models import GC_IAN1, GC_IAN2, GC_IAN3, GC_EMBED1, GC_EMBED2, GC_EMBED3, IGCN, IGCN_2, IGCN_3, IGCN_4, IGCN_5

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:

  def __init__(self, opt):
    self.opt = opt

    if 'bert' in opt.model_name:
      # tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
      # bert = BertModel.from_pretrained(opt.pretrained_bert_name)
      # self.model = opt.model_class(bert, opt).to(opt.device)
      pass
    else:
      tokenizer = build_tokenizer(
          fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
          max_seq_len=opt.max_seq_len,
          dat_fname='gen_data/tokenizer/{0}_tokenizer.dat'.format(opt.dataset))
      pos_tagger_train = build_pos_tagger(
          fname=opt.dataset_file['train'],
          dat_fname="gen_data/pos/pos_tagger_{}_train.dat".format(opt.dataset),
          modelfile=opt.stanford_pos_model,
          jarfile=opt.stanford_pos_jar,
          tokenizer=tokenizer)
      embedding_matrix = build_embedding_matrix(
          glove_path=opt.glove_path,
          word2idx=tokenizer.word2idx,
          embed_dim=opt.embed_dim,
          dat_fname=opt.embedding_matrix_path)
      if "embed" in opt.model_name:
        self.model = opt.model_class(embedding_matrix,
                                     pos_tagger_train.index2vec,
                                     opt).to(opt.device)
      else:
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

    pos_tagger_test = build_pos_tagger(
        fname=opt.dataset_file['test'],
        dat_fname="gen_data/pos/pos_tagger_{}_test.dat".format(opt.dataset),
        modelfile=opt.stanford_pos_model,
        jarfile=opt.stanford_pos_jar,
        tokenizer=tokenizer)
    self.trainset = ABSADataset(
        opt.dataset_file['train'],
        'gen_data/dataset/{}_train_dataset.dat'.format(opt.dataset), tokenizer,
        pos_tagger_train)
    self.testset = ABSADataset(
        opt.dataset_file['test'],
        'gen_data/dataset/{}_test_dataset.dat'.format(opt.dataset), tokenizer,
        pos_tagger_test)

    assert 0 <= opt.valset_ratio < 1
    if opt.valset_ratio > 0:
      valset_len = int(len(self.trainset) * opt.valset_ratio)
      self.trainset, self.valset = random_split(
          self.trainset, (len(self.trainset) - valset_len, valset_len))
    else:
      self.valset = self.testset

    if opt.device.type == 'cuda':
      logger.info('cuda memory allocated: {}'.format(
          torch.cuda.memory_allocated(device=opt.device.index)))
    self.tokenizer = tokenizer
    self._print_args()

  def _print_args(self):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in self.model.parameters():
      n_params = torch.prod(torch.tensor(p.shape))
      if p.requires_grad:
        n_trainable_params += n_params
      else:
        n_nontrainable_params += n_params
    logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(
        n_trainable_params, n_nontrainable_params))
    logger.info('> training arguments:')
    for arg in vars(self.opt):
      logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

  def _reset_params(self):
    for child in self.model.children():
      # if type(child) != BertModel:  # skip bert params
      for p in child.parameters():
        if p.requires_grad:
          if len(p.shape) > 1:
            self.opt.initializer(p)
          else:
            stdv = 1. / math.sqrt(p.shape[0])
            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

  def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
    max_val_acc = 0
    max_val_f1 = 0
    global_step = 0
    path = None
    for epoch in range(self.opt.num_epoch):
      logger.info('>' * 100)
      logger.info('epoch: {}'.format(epoch))
      n_correct, n_total, loss_total = 0, 0, 0
      # switch model to training mode
      self.model.train()
      for i_batch, sample_batched in enumerate(train_data_loader):
        global_step += 1
        # clear gradient accumulators
        optimizer.zero_grad()
        inputs = [
            sample_batched[col].to(self.opt.device)
            for col in self.opt.inputs_cols
        ]
        if self.opt.eval:
          outputs, _, _ = self.model(inputs)
        else:
          outputs = self.model(inputs)
        targets = sample_batched['polarity'].to(self.opt.device)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
        n_total += len(outputs)
        loss_total += loss.item() * len(outputs)
        if global_step % self.opt.log_step == 0:
          train_acc = n_correct / n_total
          train_loss = loss_total / n_total
          logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

      val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
      logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
      logger.info('> best_val_acc: {:.4f}, best_val_f1: {:.4f}'.format(
          max_val_acc, max_val_f1))
      if val_acc > max_val_acc:
        max_val_acc = val_acc
        if not os.path.exists('state_dict'):
          os.mkdir('state_dict')
        if self.opt.model_name == 'igcn_embed':
          path = 'state_dict/{0}_{1}_{2}_val_acc{3}'.format(
              self.opt.model_name, self.opt.model_ver, self.opt.dataset,
              round(val_acc, 4))
        else:
          path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name,
                                                        self.opt.dataset,
                                                        round(val_acc, 4))
        torch.save(self.model.state_dict(), path)
        logger.info('>> saved: {}'.format(path))
      if val_f1 > max_val_f1:
        max_val_f1 = val_f1

    return path

  def _evaluate_acc_f1(self, data_loader):
    n_correct, n_total = 0, 0
    t_inputs_all = None
    t_targets_all, t_outputs_all = None, None
    cg_all_1, cg_all_3, ag_all_1, ag_all_3 = None, None, None, None
    # switch model to evaluation mode
    self.model.eval()
    with torch.no_grad():
      for t_batch, t_sample_batched in enumerate(data_loader):
        t_inputs = [
            t_sample_batched[col].to(self.opt.device)
            for col in self.opt.inputs_cols
        ]
        t_targets = t_sample_batched['polarity'].to(self.opt.device)
        if self.opt.eval:
          t_outputs, context_gate_scores, aspect_gate_scores = self.model(
              t_inputs)
        else:
          t_outputs = self.model(t_inputs)

        n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
        n_total += len(t_outputs)

        if t_targets_all is None:
          t_targets_all = t_targets
          t_outputs_all = t_outputs
          if self.opt.eval:
            t_inputs_all = t_inputs[0]
            t_aspect_all = t_inputs[1]
            cg_all_1 = context_gate_scores[0]
            cg_all_3 = context_gate_scores[1]
            ag_all_1 = aspect_gate_scores[0]
            ag_all_3 = aspect_gate_scores[1]
        else:
          t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
          t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
          if self.opt.eval:
            t_inputs_all = torch.cat((t_inputs_all, t_inputs[0]), dim=0)
            t_aspect_all = torch.cat((t_aspect_all, t_inputs[1]), dim=0)
            cg_all_1 = torch.cat((cg_all_1, context_gate_scores[0]), dim=0)
            ag_all_1 = torch.cat((ag_all_1, aspect_gate_scores[0]), dim=0)
            cg_all_3 = torch.cat((cg_all_3, context_gate_scores[1]), dim=0)
            ag_all_3 = torch.cat((ag_all_3, aspect_gate_scores[1]), dim=0)

    acc = n_correct / n_total
    f1 = metrics.f1_score(
        t_targets_all.cpu(),
        torch.argmax(t_outputs_all, -1).cpu(),
        labels=[0, 1, 2],
        average='macro')
    if self.opt.eval:
      input_seq = t_inputs_all.cpu().numpy()
      input_text = [self.tokenizer.sequence_to_text(sq) for sq in input_seq]
      aspect_seq = t_aspect_all.cpu().numpy()
      aspect_text = [self.tokenizer.sequence_to_text(sq) for sq in aspect_seq]
      targets = t_targets_all.cpu().numpy()
      outputs = torch.argmax(t_outputs_all, -1).cpu().numpy()
      cg_all_1 = cg_all_1.cpu().numpy()
      ag_all_1 = ag_all_1.cpu().numpy()
      cg_all_3 = cg_all_3.cpu().numpy()
      ag_all_3 = ag_all_3.cpu().numpy()

      string_to_write = ""
      for i in range(0, len(input_text)):
        outstr = ''
        outstr += input_text[i] + "\n"
        outstr += aspect_text[i] + "\n"
        outstr += "Predicted: " + str(outputs[i]) + "\t" + " Actual: " + str(
            targets[i]) + '\t' + '\n\n'
        cg1 = "\t".join([str(x) for x in cg_all_1[i]]) + '\n'
        # cg3 = '\t'.join([str(x) for x in cg_all_3[i]]) + '\n'
        ag1 = "\t".join([str(x) for x in ag_all_1[i]]) + '\n'
        # ag3 = '\t'.join([str(x) for x in ag_all_3[i]]) + '\n'

        outstr += cg1 + ag1 + '\n\n'
        string_to_write += outstr
      with open('analyze_igcn_' + self.opt.dataset + '.txt', 'w') as f:
        f.write(string_to_write)

    return acc, f1

  def run(self):
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    _params = filter(lambda p: p.requires_grad, self.model.parameters())
    optimizer = self.opt.optimizer(
        _params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

    train_data_loader = DataLoader(
        dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
    test_data_loader = DataLoader(
        dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
    val_data_loader = DataLoader(
        dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
    torch.cuda.empty_cache()
    self._reset_params()
    best_model_path = self._train(criterion, optimizer, train_data_loader,
                                  val_data_loader)
    self.model.load_state_dict(torch.load(best_model_path))
    self.model.eval()
    test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
    logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(
        test_acc, test_f1))


def main():
  # Hyper Parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', default=config.MODEL_NAME, type=str)
  parser.add_argument('--model_ver', default=config.MODEL_VER, type=str)
  parser.add_argument(
      '--dataset',
      default=config.DATASET,
      type=str,
      help='twitter, restaurant, laptop')
  parser.add_argument('--optimizer', default=config.OPTIMIZER, type=str)
  parser.add_argument('--initializer', default=config.INITIALIZER, type=str)
  parser.add_argument(
      '--learning_rate',
      default=config.LEARNING_RATE,
      type=float,
      help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
  parser.add_argument('--momentum', default=config.MOMENTUM, type=float)
  parser.add_argument('--dropout', default=config.DROPOUT, type=float)
  parser.add_argument('--l2reg', default=config.L2REG, type=float)
  parser.add_argument(
      '--num_epoch',
      default=config.NUM_EPOCH,
      type=int,
      help='try larger number for non-BERT models')
  parser.add_argument(
      '--batch_size',
      default=config.BATCH_SIZE,
      type=int,
      help='try 16, 32, 64 for BERT models')
  parser.add_argument('--log_step', default=config.LOG_STEP, type=int)
  parser.add_argument('--embed_dim', default=config.EMBEDDING_DIM, type=int)
  parser.add_argument('--hidden_dim', default=config.HIDDEN_DIM, type=int)
  parser.add_argument('--bert_dim', default=768, type=int)
  parser.add_argument(
      '--pretrained_bert_name', default='bert-base-uncased', type=str)
  parser.add_argument('--embeddings', default=config.EMBEDDINGS, type=str)
  parser.add_argument('--max_seq_len', default=config.MAX_SEQ_LEN, type=int)
  parser.add_argument('--polarities_dim', default=config.POLARITY_DIM, type=int)
  parser.add_argument('--hops', default=3, type=int)
  parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
  parser.add_argument(
      '--seed',
      default=config.SEED,
      type=int,
      help='set seed for reproducibility')
  parser.add_argument(
      '--valset_ratio',
      default=config.VALSET_RATIO,
      type=float,
      help='set ratio between 0 and 1 for validation support')
  parser.add_argument('--num_layers', default=config.NUM_LAYERS, type=int)
  parser.add_argument('--in_channels', default=config.IN_CHANNELS, type=int)
  parser.add_argument('--out_channels', default=config.OUT_CHANNELS, type=int)
  parser.add_argument('--downbot', default=config.DOWNBOT, type=int)
  parser.add_argument('--kernel_size', default=config.KERNEL_SIZE, type=int)
  parser.add_argument('--position_dim', default=config.POSITION_DIM, type=int)
  parser.add_argument('--gating', default=config.GATING, type=str)
  parser.add_argument('--eval', default=config.EVAL, type=bool)
  opt = parser.parse_args()

  if opt.seed != 0:
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  emebeddings = {
      'g':
          'gen_data/embeddings/{0}_{1}_embedding_matrix.dat'.format(
              str(opt.embed_dim), opt.dataset),
      'd':
          'gen_data/embeddings/{0}_{1}_domain_embedding_matrix.dat'.format(
              str(opt.embed_dim), opt.dataset),
      'gd':
          'gen_data/embeddings/{0}_{1}_glove_domain_embedding_matrix.dat'
          .format(str(opt.embed_dim), opt.dataset)
  }
  model_classes = {
      'lstm': LSTM,
      'td_lstm': TD_LSTM,
      'tc_lstm': TC_LSTM,
      'at_lstm': AT_LSTM,
      'atae_lstm': ATAE_LSTM,
      'ian': IAN,
      'aoa': AOA,
      'gcnn': Gated_CNN,
      'gcae': GCAE,
      'gc_ian': {
          '1': GC_IAN1,
          '2': GC_IAN2,
          '3': GC_IAN3
      },
      'gc_embed': {
          '1': GC_EMBED1,
          '2': GC_EMBED2,
          '3': GC_EMBED3
      },
      'igcn_embed': {
          '1': IGCN,  # Final
          '2': IGCN_2,  # Aspect2Context
          '3': IGCN_3,  # Context2Aspect
          '4': IGCN_4,  # Context + Aspect
          '5': IGCN_5  # Context
      }
  }

  dataset_files = {
      'twitter': {
          'train': './datasets/acl-14-short-data/train.raw',
          'test': './datasets/acl-14-short-data/test.raw'
      },
      'restaurant': {
          'train': './datasets/semeval14/Restaurants_Train.xml.seg',
          'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
      },
      'laptop': {
          'train': './datasets/semeval14/Laptops_Train.xml.seg',
          'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
      }
  }
  input_colses = {
      'lstm': ['text_raw_indices'],
      'td_lstm': [
          'text_left_with_aspect_indices', 'text_right_with_aspect_indices'
      ],
      'tc_lstm': [
          'text_left_with_aspect_indices', 'text_right_with_aspect_indices',
          'aspect_indices'
      ],
      'at_lstm': ['text_raw_indices', 'aspect_indices'],
      'atae_lstm': ['text_raw_indices', 'aspect_indices'],
      'ian': ['text_raw_indices', 'aspect_indices'],
      'aoa': ['text_raw_indices', 'aspect_indices'],
      'gcnn': ['text_raw_indices'],
      'gcae': ['text_raw_indices', 'aspect_indices'],
      'gc_ian': ['text_raw_indices', 'aspect_indices'],
      'gc_embed': [
          'text_raw_indices', 'aspect_indices', 'pos_indices',
          'aspect_pos_indices', 'position_indices'
      ],
      'igcn_embed': [
          'text_raw_indices', 'aspect_indices', 'pos_indices',
          'aspect_pos_indices', 'position_indices'
      ]
  }

  initializers = {
      'xavier_uniform_': torch.nn.init.xavier_uniform_,
      'xavier_normal_': torch.nn.init.xavier_normal,
      'orthogonal_': torch.nn.init.orthogonal_,
  }
  optimizers = {
      'adadelta': torch.optim.Adadelta,  # default lr=1.0
      'adagrad': torch.optim.Adagrad,  # default lr=0.01
      'adam': torch.optim.Adam,  # default lr=0.001
      'adamax': torch.optim.Adamax,  # default lr=0.002
      'asgd': torch.optim.ASGD,  # default lr=0.01
      'rmsprop': torch.optim.RMSprop,  # default lr=0.01
      'sgd': torch.optim.SGD,
  }
  # Pos Tagger Variables
  stanford_dir = "C:/NLP_Programs/stanford-postagger-2018-10-16"
  modelfile = stanford_dir + "/models/english-bidirectional-distsim.tagger"
  jarfile = stanford_dir + "/stanford-postagger.jar"

  opt.stanford_pos_model = modelfile
  opt.stanford_pos_jar = jarfile

  opt.glove_path = './embeddings/glove.840B.300d.txt'
  opt.embedding_matrix_path = emebeddings[opt.embeddings]
  opt.model_class = model_classes[opt.model_name]
  if type(opt.model_class) == dict:
    opt.model_class = opt.model_class[opt.model_ver]
  opt.dataset_file = dataset_files[opt.dataset]
  opt.inputs_cols = input_colses[opt.model_name]
  opt.initializer = initializers[opt.initializer]
  opt.optimizer = optimizers[opt.optimizer]
  opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
      if opt.device is None else torch.device(opt.device)

  if not os.path.exists('logs'):
    os.mkdir('logs')
  if opt.model_name == 'igcn_embed':
    log_file = 'logs/{}-{}-{}-{}.log'.format(
        opt.model_name, opt.model_ver, opt.dataset,
        strftime("%y%m%d-%H%M", localtime()))
  else:
    log_file = 'logs/{}-{}-{}.log'.format(opt.model_name, opt.dataset,
                                          strftime("%y%m%d-%H%M", localtime()))
  logger.addHandler(logging.FileHandler(log_file))

  ins = Instructor(opt)
  ins.run()


if __name__ == '__main__':
  main()
