import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import IPython as ipy
import pickle
import argparse
from sklearn.utils import shuffle
import argparse
from nltk import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer
from nltk.translate.bleu_score import corpus_bleu

from model import InterpNet

######### Argument Parsing #######
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to data')
parser.add_argument('--experiment', required=True, help='experiment to evaluate')

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


# load test data
with open(os.path.join(opt.dataroot, 'test_no_cub.txt'), 'rb') as f:
    test = [line.rstrip().decode('utf-8') for line in f.readlines()]
    test_indices = [name_to_index.get(f) for f in test]

X_img_test, y_img_test, X_sentence_test, y_sentence_test, lengths_test = X_img[test_indices, :], y_img[test_indices], X_sentence[test_indices], y_sentence[test_indices], lengths[test_indices]

num_test = X_sentence_test.shape[0]

print ("Num test:", num_test)

######## LOAD MODEL #########
model_path = os.path.join(opt.experiment, 'my-model')
net_params = pickle.load(open(os.path.join(opt.experiment, 'net-params.pkl'), 'rb'))

nn = InterpNet(**net_params)
nn.load(model_path)
print ("Done loading Model")

detokenizer = MosesDetokenizer()

######### WRITE EXPLANATIONS #########

with open(os.path.join(opt.experiment, 'reference.txt'), 'wb') as f:
    for i in range(y_sentence_test.shape[0]):
        for j in range(y_sentence_test.shape[1]):
            sent_ids = y_sentence_test[i,j,:][:lengths_test[i,j]]
            sent = [id_to_word.get(s) for s in sent_ids]
            sent = detokenizer.detokenize(sent, return_str = True)
            f.write((sent+'\n').encode('utf-8'))

with open(os.path.join(opt.experiment, 'explanations.txt'), 'wb') as f:
    for i in range(X_img_test.shape[0]):
        sentence, indices = nn.get_explanation_nobeam(X_img_test[i, :][None])
        sent = [id_to_word.get(s) for s in indices]
        sent = detokenizer.detokenize(sent, return_str = True)
        f.write((sent+'\n').encode('utf-8'))
        print (str(i) + ":", sent)