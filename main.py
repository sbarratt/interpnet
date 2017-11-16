import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pickle
import argparse

import numpy as np
import IPython as ipy

from model import InterpNet

######## ARGUMENT PARSING ##########
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to folder containing dataset')
parser.add_argument('--experiment', required=True, help='path to save experiment data')
parser.add_argument('--captioning', action='store_true', help='sets model to captioning')
parser.add_argument('--output_only', action='store_true', help='sets model to output_only')
parser.add_argument('--nodropout', action='store_false', help='disables dropout')
parser.add_argument('--lrC', type=float, default=1e-3, help='learning rate for classifier, default=1e-3')
parser.add_argument('--lrE', type=float, default=5e-4, help='learning rate for explainer, default=5e-4')
parser.add_argument('--beam_width', type=int, default=1)
parser.add_argument('--num_epochs_classifier', type=int, default=100)
parser.add_argument('--num_epochs_explainer', type=int, default=100)
parser.add_argument('--batch_size_explanation', type=int, default=32)
parser.add_argument('--batch_size_classifier', type=int, default=32)
parser.add_argument('--num_hidden_lstm', type=int, default=512)
parser.add_argument('--embedding_size', type=int, default=256)
parser.add_argument('--len_norm_coeff', type=float, default=.7)
parser.add_argument('--hiddenlayers', type=int, default=1)

opt = parser.parse_args()
print(opt)

######## LOAD DATA ##########
data_dict = pickle.load(open(os.path.join(opt.dataroot, 'bird_data_dict.pkl'), 'rb'))
print ('Done loading data_dict')

X_sentence = np.array(data_dict['X_sentence'], dtype=np.int32) # (11788, 10, 76) numpy array of word IDS
y_sentence = np.array(data_dict['y_sentence'], dtype=np.int32) # (11788, 10, 76) numpy array of next word IDS, or X[:,:,1:]
lengths = np.array(data_dict['lengths'], dtype=np.int32) # (11788, 10) numpy array of lengths
vocab_size = data_dict['num_words'] # no. words in vocab
id_to_word = data_dict['id_to_word'] # dict: integer id to english word
word_to_id = data_dict['word_to_id'] # dict: english word to id
X_img = data_dict['X_img'] # (11788, 8192) numpy array of feature vectors
y_img = data_dict['y_img'] # (11788, ) numpy array of classes
max_length = data_dict['max_length'] # max length of sentence, or X_sentence.shape[-1]
print ('Done extracting objects from data_dict')

id_to_class = pickle.load(open(os.path.join(opt.dataroot, 'id_to_class.pkl'), 'rb'))
names = pickle.load(open(os.path.join(opt.dataroot, 'names.pkl'), 'rb'))
name_to_index = {} # dict: name of file to index in dataset
for i, n in enumerate(names):
    name_to_index[n] = i
print ('Done loading id_to_class and name_to_index')

######## TRAIN/VAL SPLIT ##########
with open(os.path.join(opt.dataroot, 'train_no_cub.txt'), 'rb') as f:
    train = [line.rstrip().decode('utf-8') for line in f.readlines()]
    train_indices = [name_to_index.get(f) for f in train]
with open(os.path.join(opt.dataroot, 'val_no_cub.txt'), 'rb') as f:
    val = [line.rstrip().decode('utf-8') for line in f.readlines()]
    val_indices = [name_to_index.get(f) for f in val]

train_val_overlap = bool(set(val_indices) & set(train_indices))
assert not train_val_overlap

X_img_train, y_img_train, X_sentence_train, y_sentence_train, lengths_train = X_img[train_indices, :], y_img[train_indices], X_sentence[train_indices], y_sentence[train_indices], lengths[train_indices]
X_img_val, y_img_val, X_sentence_val, y_sentence_val, lengths_val = X_img[val_indices, :], y_img[val_indices], X_sentence[val_indices], y_sentence[val_indices], lengths[val_indices]
print ("Done splitting into train/validation")

num_train = X_sentence_train.shape[0]
num_validation = X_img_val.shape[0]

print ("Num train:", num_train)
print ("Num val:", num_validation, "\n")

######## Expand Descriptions ########
# 10 descriptions per image, flatten descriptions
X_sentence_train = X_sentence_train.reshape(X_sentence_train.shape[0]*X_sentence_train.shape[1], max_length)
X_sentence_val = X_sentence_val.reshape(X_sentence_val.shape[0]*X_sentence_val.shape[1], max_length)

y_sentence_train = y_sentence_train.reshape(y_sentence_train.shape[0]*y_sentence_train.shape[1], max_length)
y_sentence_val = y_sentence_val.reshape(y_sentence_val.shape[0]*y_sentence_val.shape[1], max_length)

lengths_train = lengths_train.flatten()
lengths_val = lengths_val.flatten()

# Repeat image 10 times to match description flattening
X_img_exp_train = np.tile(X_img_train, 10).reshape(X_img_train.shape[0]*10, X_img_train.shape[1])
X_img_exp_val = np.tile(X_img_val, 10).reshape(X_img_val.shape[0]*10, X_img_val.shape[1])
print ("Done Expanding descriptions\n")

os.makedirs(opt.experiment, exist_ok = True)


num_in = X_img.shape[1]
if opt.hiddenlayers == 1:
    num_hiddens = [500]
elif opt.hiddenlayers == 2:
    num_hiddens = [500, 300]
else:
    assert 0
num_out = len(id_to_class.keys())

net_params = {
    'num_in': num_in,
    'num_hiddens': num_hiddens,
    'num_out': num_out,
    'batch_size_classifier': opt.batch_size_classifier,
    'lr_classifier': opt.lrC,
    'embedding_size': opt.embedding_size,
    'num_hidden_lstm': opt.num_hidden_lstm,
    'vocab_size': vocab_size,
    'max_length': max_length,
    'batch_size_explanation': opt.batch_size_explanation,
    'lr_explanation': opt.lrE,
    'beam_width': opt.beam_width,
    'len_norm_coeff': opt.len_norm_coeff,
    'num_epochs_classifier': opt.num_epochs_classifier,
    'num_epochs_explainer': opt.num_epochs_explainer,
    'id_to_word': id_to_word,
    'word_to_id': word_to_id,
    'scope': str(i),
    'dropout': opt.nodropout,
    'captioning': opt.captioning,
    'output_only': opt.output_only
}
pickle.dump(net_params, open(os.path.join(opt.experiment, 'net-params.pkl'), 'wb'))

nn = InterpNet(**net_params)
print ("Done Initializing InterpNet")

nn.fit(
    X_img_train,
    y_img_train,
    X_img_exp_train,
    X_img_exp_val,
    X_sentence_train,
    y_sentence_train,
    lengths_train,
    X_img_val,
    y_img_val,
    X_sentence_val,
    y_sentence_val,
    lengths_val,
    num_validation,
    opt.experiment
)

print ("Done Training")

