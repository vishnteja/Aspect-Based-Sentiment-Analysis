import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from nltk.tag import StanfordPOSTagger

java_path = "C:/Program Files/Java/jdk1.8.0_181/bin/java.exe"
os.environ["JAVAHOME"] = java_path
stanford_dir = "C:/NLP_Programs/stanford-postagger-2018-10-16"
modelfile = stanford_dir + "/models/english-bidirectional-distsim.tagger"
jarfile = stanford_dir + "/stanford-postagger.jar"

tagger = StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)

pos2index = {
        "CC": 1,
        "CD": 2,
        "DT": 3,
        "EX": 4,
        "FW": 5,
        "IN": 6,
        "JJ": 7,
        "JJR": 8,
        "JJS": 9,
        "LS": 10,
        "MD": 11,
        "NN": 12,
        "NNS": 13,
        "NP": 14,
        "NNPS": 15,
        "PDT": 16,
        "POS": 17,
        "PRP": 18,
        "PRP$": 19,
        "RB": 20,
        "RBR": 21,
        "RBS": 22,
        "RP": 23,
        "SYM": 24,
        "TO": 25,
        "UH": 26,
        "VB": 27,
        "VBG": 29,
        "VBD": 28,
        "VBN": 30,
        "VBP": 31,
        "VBZ": 32,
        "WDT": 33,
        "WP": 34,
        "WP$": 35,
        "WRB": 36
}

def word_tokenize_text(text):
  for ch in [
      "\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*",
      "/", "?", "(", ")", "\"", "-", ":"
  ]:
    text = text.replace(ch, " " + ch + " ")
  return text

def build_pos_tagger(modelfile, jarfile):
  return POSTagger(modelfile, jarfile)

def build_tokenizer(fnames, max_seq_len, dat_fname):
  if os.path.exists(dat_fname):
    print('loading tokenizer:', dat_fname)
    tokenizer = pickle.load(open(dat_fname, 'rb'))
  else:
    text = ''
    for fname in fnames:
      fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
      lines = fin.readlines()
      fin.close()
      for i in range(0, len(lines), 3):
        text_left, _, text_right = [
            s.lower().strip() for s in lines[i].partition("$T$")
        ]
        aspect = lines[i + 1].lower().strip()
        text_raw = text_left + " " + aspect + " " + text_right
        text += word_tokenize_text(text_raw) + " "

    tokenizer = Tokenizer(max_seq_len)
    tokenizer.fit_on_text(text)
    pickle.dump(tokenizer, open(dat_fname, 'wb'))
  return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
  fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
  word_vec = {}
  for line in fin:
    tokens = line.rstrip().split()
    if word2idx is None or tokens[0] in word2idx.keys():
      word_vec[tokens[0]] = np.asarray(tokens[-embed_dim:], dtype='float32')
  return word_vec


def build_embedding_matrix(glove_path, word2idx, embed_dim, dat_fname):
  if os.path.exists(dat_fname):
    print('loading embedding_matrix:', dat_fname)
    embedding_matrix = pickle.load(open(dat_fname, 'rb'))
  else:
    print('loading word vectors...')
    embedding_matrix = np.zeros(
        (len(word2idx) + 2,
         embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
    fname = glove_path
    word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
    print('building embedding_matrix:', dat_fname)
    for word, i in word2idx.items():
      vec = word_vec.get(word)
      if vec is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = vec
    pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
  return embedding_matrix


def pad_and_truncate(sequence,
                     maxlen,
                     dtype='int64',
                     padding='post',
                     truncating='post',
                     value=0):
  x = (np.ones(maxlen) * value).astype(dtype)
  if truncating == 'pre':
    trunc = sequence[-maxlen:]
  else:
    trunc = sequence[:maxlen]
  trunc = np.asarray(trunc, dtype=dtype)
  if padding == 'post':
    x[:len(trunc)] = trunc
  else:
    x[-len(trunc):] = trunc
  return x


class Tokenizer(object):

  def __init__(self, max_seq_len, lower=True):
    self.lower = lower
    self.max_seq_len = max_seq_len
    self.word2idx = {}
    self.idx2word = {}
    self.idx = 1

  def fit_on_text(self, text):
    if self.lower:
      text = text.lower()
    text = word_tokenize_text(text)
    words = text.split()
    for word in words:
      if word not in self.word2idx:
        self.word2idx[word] = self.idx
        self.idx2word[self.idx] = word
        self.idx += 1

  def text_to_sequence(self,
                       text,
                       reverse=False,
                       padding='post',
                       truncating='post'):
    if self.lower:
      text = text.lower()
    text = word_tokenize_text(text)
    words = text.split()
    unknownidx = len(self.word2idx) + 1
    sequence = [
        self.word2idx[w] if w in self.word2idx else unknownidx for w in words
    ]
    if len(sequence) == 0:
      sequence = [0]
    if reverse:
      sequence = sequence[::-1]
    return pad_and_truncate(
        sequence, self.max_seq_len, padding=padding, truncating=truncating)

  def position_to_sequence(self, text_left, aspect, text_right):
    tag_left = [len(text_left)-i for i in range(len(text_left))]
    tag_aspect = [0 for i in range(len(text_aspect))]
    tag_right = [i+1 for i in range(len(text_right))]
    position_tag = tag_left + tag_aspect + tag_right
    if len(position_tag) == 0:
      position_tag = [0]

    return pad_and_truncate(
      sequence, self.max_seq_len, padding=padding, truncating=truncating)

class POSTagger():

  def __init__(self, modelfile, jarfile):
    self.tagger = StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)
    self.pos2index = pos2index
    self.num_tags = len(self.pos2index)
    self.index2vec = np.zeros((self.num_tags+2, self.num_tags)

    for i in range(0, self.num_tags):
      self.index2vec[i+1] = np.zeros(self.num_tags)
      self.index2vec[i+1, i] = 1

  def get_pos_tags(text):
    tagged_text = self.tagger.tag(text)
    tags = []
    for i in tagged_text:
      _, tag = zip(*i)
      if tag in self.pos2index:
        tags.append(self.pos2index[tag])
      else:
        tags.append(self.pos2index["NOT$"])
    return tags


class ABSADataset(Dataset):

  def __init__(self, fname, tokenizer, pos_tagger):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()

    all_data = []
    for i in range(0, len(lines), 3):
      text_left, _, text_right = [
          s.lower().strip() for s in lines[i].partition("$T$")
      ]
      text_left = word_tokenize_text(text_left)
      text_right = word_tokenize_text(text_right)
      aspect = word_tokenize_text(lines[i + 1].lower().strip())
      polarity = lines[i + 2].strip()

      text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect +
                                                    " " + text_right)
      text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left +
                                                                   " " +
                                                                   text_right)
      text_left_indices = tokenizer.text_to_sequence(text_left)
      text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left +
                                                                 " " + aspect)
      text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
      text_right_with_aspect_indices = tokenizer.text_to_sequence(
          " " + aspect + " " + text_right, reverse=True)
      aspect_indices = tokenizer.text_to_sequence(aspect)
      left_context_len = np.sum(text_left_indices != 0)
      aspect_len = np.sum(aspect_indices != 0)
      aspect_in_text = torch.tensor(
          [left_context_len.item(), (left_context_len + aspect_len - 1).item()])
      polarity = int(polarity) + 1

      text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left +
                                                     " " + aspect + " " +
                                                     text_right + ' [SEP] ' +
                                                     aspect + " [SEP]")
      bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) +
                                     [1] * (aspect_len + 1))
      bert_segments_ids = pad_and_truncate(bert_segments_ids,
                                           tokenizer.max_seq_len)

      text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left +
                                                         " " + aspect + " " +
                                                         text_right + " [SEP]")
      aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect +
                                                       " [SEP]")

      pos_indices = pos_tagger.get_pos_tags(text_left + " " + aspect +
                                                    " " + text_right)

      position_indices = tokenizer.position_to_sequence(text_left, aspect, text_right)

      data = {
          'text_bert_indices': text_bert_indices,
          'bert_segments_ids': bert_segments_ids,
          'text_raw_bert_indices': text_raw_bert_indices,
          'aspect_bert_indices': aspect_bert_indices,
          'text_raw_indices': text_raw_indices,
          'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
          'text_left_indices': text_left_indices,
          'text_left_with_aspect_indices': text_left_with_aspect_indices,
          'text_right_indices': text_right_indices,
          'text_right_with_aspect_indices': text_right_with_aspect_indices,
          'aspect_indices': aspect_indices,
          'aspect_in_text': aspect_in_text,
          'polarity': polarity,
          'pos_indices': pos_indices
      }

      all_data.append(data)
    self.data = all_data

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)
