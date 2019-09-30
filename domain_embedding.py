import pickle
import numpy as np
from data_utils import build_tokenizer
domain = 'laptop'
tokenizer_path = './gen_data/tokenizer/{}_tokenizer.dat'.format(domain)
word_vec_path = './embeddings/domain_embedding/{}_emb.vec'.format(domain)
glove_path = './embeddings/glove.840B.300d.txt'
out_path_d = './gen_data/embeddings/100_{}_domain_embedding_matrix.dat'.format(
    domain)
out_path_gd = './gen_data/embeddings/400_{}_glove_domain_embedding_matrix.dat'.format(
    domain)


def _load_word_vec(path, word2idx=None, embed_dim=300):
  fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
  word_vec = {}
  for line in fin:
    tokens = line.rstrip().split()
    if word2idx is None or tokens[0] in word2idx.keys():
      word_vec[tokens[0]] = np.asarray(tokens[-embed_dim:], dtype='float32')
  return word_vec


tokenizer = pickle.load(open(tokenizer_path, 'rb'))
word2idx = tokenizer.word2idx
word_vec_d = _load_word_vec(word_vec_path, word2idx=word2idx, embed_dim=100)
word_vec_g = _load_word_vec(glove_path, word2idx=word2idx, embed_dim=300)

embedding_matrix_d = np.zeros(
    (len(word2idx) + 2, 100))  # idx 0 and len(word2idx)+1 are all-zeros
embedding_matrix_gd = np.zeros((len(word2idx) + 2, 400))

for word, i in word2idx.items():
  vec = word_vec_d.get(word)
  glo = word_vec_g.get(word)
  if vec is not None and glo is not None:
    embedding_matrix_d[i] = vec
    embedding_matrix_gd[i] = np.concatenate((glo, vec))

pickle.dump(embedding_matrix_d, open(out_path_d, 'wb'))
pickle.dump(embedding_matrix_gd, open(out_path_gd, 'wb'))