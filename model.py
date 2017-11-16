"""
model.py

Defines InterpNet class which generates classifications and explanations and trains on supervised classification/explanation data.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import pickle
import time
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import math
import IPython as ipy
from nltk.tokenize.moses import MosesDetokenizer

# normc weight initializer from CS294-112
def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class InterpNet(object):
    """
    Class for InterpNet. Classifies Inceptionv3-preprocessed images into classes with one hidden layer and then generates a natural language explanation of the classification.

    num_in: dimensionality of input
    num_hiddens: list of hidden unit sizes
    num_out: number of output neurons
    batch_size_classifier: batch size for classifier SGD
    lr_classifier: learning rate for classifier AdamOptimizer
    embedding_size: dimensionality of word embedding
    num_hidden_lstm: number of hidden LSTM units
    vocab_size: number of words in vocabulary
    max_length: maximum length of sentence
    batch_size_explanation: batch size for explanation AdamOptimizer
    lr_explanation: learning rate for explanation AdamOptimizer
    beam_width: beam width for beam search LSTM unrolling
    len_norm_coeff: Length normalization constant used in beam search LSTM unrolling
    num_epochs_classifier: number of epochs to train for
    num_epochs_explainer: number of epochs to train for
    id_to_word: dictionary mapping integer IDs to english words strings
    word_to_id: dictionary mapping english word strings to integer IDs
    scope: string for scope of all tensorflow variables
    """
    def __init__(self,
            num_in = 2048,
            num_hiddens = [250],
            num_out = 200,
            batch_size_classifier = 50,
            lr_classifier = 1e-3,
            embedding_size = 50,
            num_hidden_lstm = 128,
            vocab_size = 100,
            max_length = 100,
            batch_size_explanation = 50,
            lr_explanation = 1e-3,
            beam_width = 7,
            len_norm_coeff = 0.7,
            num_epochs_classifier = 5,
            num_epochs_explainer = 20,
            id_to_word = None,
            word_to_id = None,
            scope = '0',
            dropout=False,
            captioning=False,
            output_only=False):

        ################## Classifier ##################
        self.num_in = num_in
        self.num_hiddens = num_hiddens
        self.num_out = num_out
        self.batch_size_classifier = batch_size_classifier
        self.lr_classifier = lr_classifier
        self.num_epochs_classifier = num_epochs_classifier

        #################### LSTM ####################
        self.embedding_size = embedding_size
        self.num_hidden_lstm = num_hidden_lstm
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size_explanation = batch_size_explanation
        self.lr_explanation = lr_explanation
        self.beam_width = beam_width
        self.len_norm_coeff = len_norm_coeff
        self.num_epochs_explainer = num_epochs_explainer
        self.dropout = dropout
        self.captioning = captioning
        self.output_only = output_only

        ################### General ##################
        self.id_to_word = id_to_word
        self.word_to_id = word_to_id
        self.scope = scope
        self.initialized = False

        self.initialize_network()

    def initialize_network(self):
        tf.reset_default_graph()


        # Placeholders
        self.sy_X = tf.placeholder(tf.float32, [None, self.num_in]) # inputs to classifier
        self.sy_y = tf.placeholder(tf.int32, [None]) # outputs of classifier
        self.sy_labels = tf.one_hot(self.sy_y, self.num_out, axis=-1) # one-hot representation of outputs
        self.sy_lr_classifier = tf.placeholder(tf.float32) # variable learning rate

        ###### Classifier #######
        self.sy_layers = [self.sy_X]
        for i, num_hidden in enumerate(self.num_hiddens):
            self.sy_layers.append(
                tf.contrib.layers.fully_connected(
                    inputs = self.sy_layers[-1],
                    num_outputs = num_hidden,
                    activation_fn = tf.nn.relu,
                    weights_initializer = normc_initializer(1.0),
                    biases_initializer = tf.constant_initializer(.1)
                )
            )

        self.sy_logits = tf.contrib.layers.fully_connected(
            inputs = self.sy_layers[-1],
            num_outputs = self.num_out,
            activation_fn = None,
            weights_initializer = normc_initializer(1.0),
            biases_initializer = tf.constant_initializer(0.0)
        )
        self.sy_probs = tf.nn.softmax(self.sy_logits)
        self.sy_layers.append(self.sy_probs)
        self.sy_predictions = tf.argmax(self.sy_probs, axis=1)

        self.sy_classification_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels = self.sy_labels,
                logits = self.sy_logits
            )
        )
        self.sy_optimize_classifier_step = tf.train.AdamOptimizer(self.sy_lr_classifier).minimize(self.sy_classification_loss)

        ###### LSTM ######

        # Placeholders
        self.sy_batchsize_explanation = tf.placeholder(tf.int32, [])
        self.sy_seq_lengths = tf.placeholder(tf.int32, [None])
        self.sy_rnn_inputs = tf.placeholder(tf.int32, [None, self.max_length])
        self.sy_rnn_outputs = tf.placeholder(tf.int32, [None, self.max_length])
        self.rnn_outputs = tf.one_hot(self.sy_rnn_outputs, depth=self.vocab_size)
        self.sy_c_state1 = tf.placeholder(tf.float32, [None, self.num_hidden_lstm])
        self.sy_h_state1 = tf.placeholder(tf.float32, [None, self.num_hidden_lstm])
        self.sy_initial_state1 = tf.contrib.rnn.LSTMStateTuple(self.sy_c_state1, self.sy_h_state1)
        self.sy_c_state2 = tf.placeholder(tf.float32, [None, self.num_hidden_lstm])
        self.sy_h_state2 = tf.placeholder(tf.float32, [None, self.num_hidden_lstm])
        self.sy_initial_state2 = tf.contrib.rnn.LSTMStateTuple(self.sy_c_state2, self.sy_h_state2)
        self.sy_lr_explanation = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        # Random initial embedding
        self.sy_W_embedding = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0)
        )
        self.sy_input_embedding = tf.nn.embedding_lookup(self.sy_W_embedding, self.sy_rnn_inputs)

        # concatenation of features to send as input to LSTM
        if self.captioning:
            self.sy_classification_layers = tf.stop_gradient(self.sy_X)
        elif self.output_only:
            self.sy_classification_layers = tf.stop_gradient(self.sy_probs)
        else:
            self.sy_classification_layers = tf.stop_gradient(tf.concat(self.sy_layers, axis=1))

        # Network Structure
        # LSTM 1
        with tf.variable_scope('rnn1'):
            if self.dropout:
                self.sy_lstm_cell1 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_hidden_lstm, state_is_tuple=True), output_keep_prob=self.keep_prob)
            else:
                self.sy_lstm_cell1 = tf.contrib.rnn.LSTMCell(self.num_hidden_lstm, state_is_tuple=True)
            self.sy_zero_initial_state1 = self.sy_lstm_cell1.zero_state(self.sy_batchsize_explanation, tf.float32)
            self.sy_lstm_outputs1, self.sy_lstm_states1 = tf.nn.dynamic_rnn(self.sy_lstm_cell1, self.sy_input_embedding, dtype=tf.float32, sequence_length=self.sy_seq_lengths, initial_state=self.sy_initial_state1)

        self.sy_W1 = tf.Variable(normc_initializer(1.0)(
            [self.num_hidden_lstm + int(self.sy_classification_layers.get_shape()[-1]),
            self.num_hidden_lstm]))
        self.sy_b1 = tf.Variable(tf.constant_initializer(0.0)((self.num_hidden_lstm,)))

        self.hidden_projection = lambda x: tf.matmul(tf.concat([x, self.sy_classification_layers], axis=-1), self.sy_W1) + self.sy_b1

        self.sy_lstm_outputs1 = tf.transpose(self.sy_lstm_outputs1, [1, 0, 2])
        self.sy_lstm_inputs2 = tf.map_fn(self.hidden_projection, self.sy_lstm_outputs1)
        self.sy_lstm_inputs2 = tf.transpose(self.sy_lstm_inputs2, [1, 0, 2])

        # LSTM 2
        with tf.variable_scope('rnn2'):
            if self.dropout:
                self.sy_lstm_cell2 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_hidden_lstm, state_is_tuple=True), output_keep_prob=self.keep_prob)
            else:
                self.sy_lstm_cell2 = tf.contrib.rnn.LSTMCell(self.num_hidden_lstm, state_is_tuple=True)
            self.sy_zero_initial_state2 = self.sy_lstm_cell2.zero_state(self.sy_batchsize_explanation, tf.float32)
            self.sy_lstm_outputs2, self.sy_lstm_states2 = tf.nn.dynamic_rnn(self.sy_lstm_cell2, self.sy_lstm_inputs2, dtype=tf.float32, sequence_length=self.sy_seq_lengths, initial_state=self.sy_initial_state2)

        self.sy_W2 = tf.Variable(normc_initializer(1.0)(
            [self.num_hidden_lstm, self.vocab_size]))
        self.sy_b2 = tf.Variable(tf.constant_initializer(0.0)((self.vocab_size,)))

        self.logit_projection = lambda x: tf.matmul(x, self.sy_W2) + self.sy_b2

        self.sy_lstm_outputs2 = tf.transpose(self.sy_lstm_outputs2, [1, 0, 2])
        self.sy_final_logits = tf.map_fn(self.logit_projection, self.sy_lstm_outputs2)
        self.sy_final_logits = tf.transpose(self.sy_final_logits, [1, 0, 2])
        self.sy_sentence_word_probs = tf.nn.softmax(self.sy_final_logits, dim=-1)

        # Loss and Optimizer
        # mask based on self.sy_seq_lengths
        # this code is super confusing, but essentially masks all the cross entropies so that entries after seq_length for each batch are ignored in the mean cross entropy computation
        self.ones = tf.ones((self.sy_batchsize_explanation, self.max_length), dtype=tf.int32)
        self.zeros = tf.zeros((self.sy_batchsize_explanation, self.max_length), dtype=tf.int32)
        self.lengths_transposed = tf.reshape(self.sy_seq_lengths, [-1, 1])
        self.lengths_transposed = tf.tile(self.lengths_transposed, [1, self.max_length])
        self.range = tf.range(0, self.max_length, 1)
        self.range_row = tf.reshape(self.range, [-1, 1])
        self.range_row = tf.transpose(tf.tile(self.range_row, [1, self.sy_batchsize_explanation]))
        self.mask_int = tf.less(self.range_row, self.lengths_transposed)
        self.mask = tf.where(self.mask_int, self.ones, self.zeros)
        self.cross_entropy = self.rnn_outputs * tf.log(self.sy_sentence_word_probs + 1e-8)
        self.cross_entropy = -tf.reduce_sum(self.cross_entropy, reduction_indices=2)
        self.cross_entropy = self.cross_entropy * tf.cast(self.mask, tf.float32)
        self.cross_entropy = tf.reduce_sum(self.cross_entropy, reduction_indices=1)
        self.cross_entropy = self.cross_entropy / tf.cast(self.sy_seq_lengths, tf.float32)

        self.sy_explanation_loss = tf.reduce_mean(self.cross_entropy)

        self.sy_optimize_explanation_step = tf.train.AdamOptimizer(self.sy_lr_explanation).minimize(self.sy_explanation_loss)

        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.sess.run(tf.global_variables_initializer())

        self.initialized=True

    def save(self, f):
        self.saver.save(self.sess, f)

    def load(self, f):
        self.saver.restore(self.sess, f)

    def validation_loss_and_accuracy_classifier(self, X_img_val, y_img_val):
        preds = self.sess.run([self.sy_predictions], feed_dict={
                self.sy_X: X_img_val
            })[0]

        val_loss = self.sess.run([self.sy_classification_loss], feed_dict={
                self.sy_X: X_img_val,
                self.sy_y: y_img_val
            })[0]

        return val_loss, float(np.sum(preds == y_img_val)) / y_img_val.shape[0]

    def validation_loss_explanation(self, X_img_val, X_sentence_val, y_sentence_val, lengths_val):
        batch_size = self.batch_size_explanation

        c_initial1, h_initial1 = self.sess.run([self.sy_zero_initial_state1], feed_dict={
                self.sy_batchsize_explanation: batch_size
            })[0]

        c_initial2, h_initial2 = self.sess.run([self.sy_zero_initial_state2], feed_dict={
                self.sy_batchsize_explanation: batch_size
            })[0]

        num_validation = X_img_val.shape[0]
        val_loss = 0.0
        n_iter = int(num_validation / batch_size)

        for i in range(n_iter):
            feed = {
                    self.sy_X: X_img_val[i*batch_size:(i+1)*batch_size, :],
                    self.sy_rnn_inputs: X_sentence_val[i*batch_size:(i+1)*batch_size, :],
                    self.sy_rnn_outputs: y_sentence_val[i*batch_size:(i+1)*batch_size, :],
                    self.sy_seq_lengths: lengths_val[i*batch_size:(i+1)*batch_size],
                    self.sy_c_state1: c_initial1,
                    self.sy_h_state1: h_initial1,
                    self.sy_c_state2: c_initial2,
                    self.sy_h_state2: h_initial2,
                    self.sy_batchsize_explanation: batch_size,
                    self.keep_prob: 1.0
                }
            loss = self.sess.run([self.sy_explanation_loss],
            feed_dict=feed)[0]
            val_loss += loss / n_iter

        return val_loss

    def bleu_explanation(self, X_img_val, X_sentence_val, y_sentence_val, lengths_val):
        # TODO: modify for bigger reference set
        chencherry = SmoothingFunction()
        num_validation = X_img_val.shape[0]

        exps = []
        for i in range(num_validation):
            img = X_img_val[i, :]
            _, x = self.get_explanation_nobeam(img[None])
            exps.append(x)

        refs = []
        for i in range(num_validation):
            refs.append([y_sentence_val[i][:lengths_val[i]]])

        bleu = corpus_bleu(refs, exps, smoothing_function=chencherry.method0)
        return bleu

    def fit(self,
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
            folder):

        metrics = {
            'epoch_classifier': [],
            'epoch_explainer': [],
            'val_loss_classifier': [],
            'val_loss_explanation': [],
            'train_loss_classifier': [],
            'train_loss_explanation': [],
            'val_accuracy': []
        }

        model_path = os.path.join(folder, 'my-model')
        self.save(model_path)

        start = time.time()

        best_val_accuracy = 0.0
        for epoch in range(self.num_epochs_classifier):
            print ("\n------Classifier Epoch %d------" % (epoch + 1))

            train_loss_classifier = self.classifier_epoch(X_img_train, y_img_train)
            val_loss_classifier, val_accuracy = self.validation_loss_and_accuracy_classifier(X_img_val, y_img_val)
            metrics['train_loss_classifier'].append(train_loss_classifier)
            metrics['val_loss_classifier'].append(val_loss_classifier)
            metrics['val_accuracy'].append(val_accuracy)
            metrics['epoch_classifier'].append(epoch)
            print ("Train_loss: %.6f" % train_loss_classifier)
            print ("Validation_loss: %.6f" % val_loss_classifier)
            print ("Validation_accuracy: %.6f" % val_accuracy)

            if val_accuracy > best_val_accuracy:
                print ("Saving model")
                self.save(model_path)
                best_val_accuracy = val_accuracy
            else:
                self.lr_classifier /= 2 # anneal learning rate
            pickle.dump(metrics, open(os.path.join(folder, 'metrics.pkl'), 'wb'))
        self.load(model_path)
        best_val_loss_explainer = float("inf")

        for epoch in range(self.num_epochs_explainer):
            print ("\n------Explainer Epoch %d------" % (epoch + 1))

            train_loss_explanation = self.explanation_epoch(X_img_exp_train, X_sentence_train, y_sentence_train, lengths_train)
            val_loss_explanation = self.validation_loss_explanation(X_img_exp_val, X_sentence_val, y_sentence_val, lengths_val)

            metrics['train_loss_explanation'].append(train_loss_explanation)
            metrics['val_loss_explanation'].append(val_loss_explanation)
            metrics['epoch_explainer'].append(epoch)

            print ("Train_loss: %.6f" % train_loss_explanation)
            print ("Validation_loss: %.6f" % val_loss_explanation)

            if val_loss_explanation < best_val_loss_explainer:
                print ("Saving model")
                self.save(model_path)
                best_val_loss_explainer = val_loss_explanation
            else:
                self.lr_explanation /= 2 # anneal learning rate
            pickle.dump(metrics, open(os.path.join(folder, 'metrics.pkl'), 'wb'))

        self.load(model_path)

        end = time.time()
        metrics['train_time'] = end-start

        print ("Writing Metrics to file...")
        pickle.dump(metrics, open(os.path.join(folder, 'metrics.pkl'), 'wb'))

    def classifier_epoch(self, X_img_train, y_train):
        n_iter = int(X_img_train.shape[0] / self.batch_size_classifier)
        X_img_train, y_train = shuffle(X_img_train, y_train)

        epoch_loss = 0.0

        for i in range(n_iter):
            X_batch = X_img_train[i*self.batch_size_classifier:(i+1)*self.batch_size_classifier,:]
            y_batch = y_train[i*self.batch_size_classifier:(i+1)*self.batch_size_classifier]

            feed = {
                self.sy_X: X_batch,
                self.sy_y: y_batch,
                self.sy_lr_classifier: self.lr_classifier
            }
            loss, _ = self.sess.run([self.sy_classification_loss, self.sy_optimize_classifier_step], feed_dict=feed)

            epoch_loss += loss / (n_iter * self.batch_size_classifier)

        return epoch_loss

    def explanation_epoch(self, X_img_train, X_exp_train, y_train, lengths_train):
        n_iter = int(X_exp_train.shape[0] / self.batch_size_explanation)
        X_img_train, X_exp_train, y_train, lengths_train = shuffle(X_img_train, X_exp_train, y_train, lengths_train)

        epoch_loss = 0.0

        for i in range(n_iter):
            X_img_batch = X_img_train[i*self.batch_size_explanation:(i+1)*self.batch_size_explanation, :]
            X_exp_batch = X_exp_train[i*self.batch_size_explanation:(i+1)*self.batch_size_explanation, :]
            y_batch = y_train[i*self.batch_size_explanation:(i+1)*self.batch_size_explanation, :]
            lengths_batch = lengths_train[i*self.batch_size_explanation:(i+1)*self.batch_size_explanation]

            c_initial1, h_initial1 = self.sess.run([self.sy_zero_initial_state1], feed_dict={
                    self.sy_batchsize_explanation: self.batch_size_explanation
                })[0]

            c_initial2, h_initial2 = self.sess.run([self.sy_zero_initial_state2], feed_dict={
                    self.sy_batchsize_explanation: self.batch_size_explanation
                })[0]

            feed = {
                self.sy_X: X_img_batch,
                self.sy_rnn_inputs: X_exp_batch,
                self.sy_rnn_outputs: y_batch,
                self.sy_seq_lengths: lengths_batch,
                self.sy_lr_explanation: self.lr_explanation,
                self.sy_c_state1: c_initial1,
                self.sy_h_state1: h_initial1,
                self.sy_c_state2: c_initial2,
                self.sy_h_state2: h_initial2,
                self.sy_batchsize_explanation: self.batch_size_explanation,
                self.keep_prob: .8
            }

            loss, _ = self.sess.run([self.sy_explanation_loss, self.sy_optimize_explanation_step], feed_dict = feed)

            epoch_loss += loss / (n_iter)

        return epoch_loss

    def get_explanation_nobeam(self, image):
        c1, h1 = self.sess.run([self.sy_zero_initial_state1], feed_dict={
                        self.sy_batchsize_explanation: 1
                    })[0]
        c2, h2 = self.sess.run([self.sy_zero_initial_state2], feed_dict={
                        self.sy_batchsize_explanation: 1
                    })[0]

        indices = []

        k = 0
        ind = 0
        while 1:
            state1, state2, probs = self.sess.run([self.sy_lstm_states1, self.sy_lstm_states2, self.sy_sentence_word_probs],
                    feed_dict={
                        self.sy_X: image,
                        self.sy_rnn_inputs: np.array([ind] + [0]*(self.max_length-1), dtype=np.int32)[None],
                        self.sy_seq_lengths: np.array([1], dtype=np.int32),
                        self.sy_c_state1: c1,
                        self.sy_h_state1: h1,
                        self.sy_c_state2: c2,
                        self.sy_h_state2: h2,
                        self.sy_batchsize_explanation: 1,
                        self.keep_prob: 1.
                    })
            c1, h1 = state1
            c2, h2 = state2
            ind = np.argmax(probs)
            indices.append(ind)
            if self.id_to_word.get(ind) == '.':
                break

            k += 1

            if k == self.max_length:
                indices.append(self.word_to_id.get('.'))
                break

        detokenizer = MosesDetokenizer()

        return detokenizer.detokenize([self.id_to_word.get(i) for i in indices], return_str = True), indices

    def get_explanation(self, image):
        # Initialize beam
        iters = 0
        ind = 0
        c1, h1 = self.sess.run([self.sy_zero_initial_state1], feed_dict={
                        self.sy_batchsize_explanation: 1
                    })[0]
        c2, h2 = self.sess.run([self.sy_zero_initial_state2], feed_dict={
                        self.sy_batchsize_explanation: 1
                    })[0]

        state1, state2, probs = self.sess.run([self.sy_lstm_states1, self.sy_lstm_states2, self.sy_sentence_word_probs],
                        feed_dict={
                            self.sy_X: image,
                            self.sy_rnn_inputs: np.array([ind] + [0]*(self.max_length-1), dtype=np.int32)[None],
                            self.sy_seq_lengths: np.array([1], dtype=np.int32),
                            self.sy_c_state1: c1,
                            self.sy_h_state1: h1,
                            self.sy_c_state2: c2,
                            self.sy_h_state2: h2,
                            self.sy_batchsize_explanation: 1,
                            self.keep_prob: 1.0
                        })
        c1, h1 = state1
        c2, h2 = state2
        ipy.embed()
        beam = np.argsort(probs[0,0,:])[::-1][:self.beam_width]
        beam_probs = [probs[0,0,:][ind] for ind in beam]
        hypothesis = [([ind], math.log(prob), c1, h1, c2, h2) for ind, prob in zip(beam, beam_probs)]

        # Beam Search
        num_completed = 0
        seen_sentences = []
        while 1:
            iters += 1
            new_hypothesis = []
            for i, datum in enumerate(hypothesis):
                indices, prob, c1, h1, c2, h2  = datum
                last_ind = indices[-1] # Get last index used in hypothesis
                if self.id_to_word[last_ind] == ".":
                    if datum not in seen_sentences:
                        num_completed = num_completed + 1 # Keep track of effective beam width (if it reaches 0 are done)
                        seen_sentences.append(datum) # Keep track of sentences you have seen to know when to stop search
                    new_hypothesis.append(datum) # Keep in beam
                    continue
                state1, state2, probs = self.sess.run([self.sy_lstm_states1, self.sy_lstm_states2, self.sy_sentence_word_probs],
                                feed_dict={
                                    self.sy_X: image,
                                    self.sy_rnn_inputs: np.array([ind] + [0]*(self.max_length-1), dtype=np.int32)[None],
                                    self.sy_seq_lengths: np.array([1], dtype=np.int32),
                                    self.sy_c_state1: c1,
                                    self.sy_h_state1: h1,
                                    self.sy_c_state2: c2,
                                    self.sy_h_state2: h2,
                                    self.sy_batchsize_explanation: 1,
                                    self.keep_prob: 1.0
                                })
                c1, h1 = state1
                c2, h2 = state2
                beam = np.argsort(probs[0,0,:])[::-1][:self.beam_width]
                beam_probs = [probs[0,0,:][ind] for ind in beam]
                new_beam = [(indices + [ind], prob+math.log(prob_new), c1, h1, c2, h2) for ind, prob_new in zip(beam, beam_probs)]
                new_hypothesis.extend(new_beam)

            # new_hypothesis contains all complete sentences or max length reached -> return the best result
            if num_completed == self.beam_width or iters == self.max_length:
                indices, prob, c1, h1, c2, h2 = sorted(new_hypothesis, key= lambda tup: tup[1]/(math.pow((5+len(tup[0])), self.len_norm_coeff) / math.pow(6, self.len_norm_coeff)))[::-1][0]
                sentence = " ".join([self.id_to_word[ind] for ind in indices])
                if '.' in sentence:
                    sentence = sentence[:-2]+"." # Move period
                else:
                    sentence = sentence+"." # Append period
                    indices.append(self.word_to_id['.']) # Append period
                return sentence, indices

            # Take top beam_width results from new hypothesis (normalized by length)
            hypothesis = sorted(new_hypothesis, key= lambda tup: tup[1]/(math.pow((5+len(tup[0])), self.len_norm_coeff) / math.pow(6, self.len_norm_coeff)))[::-1][:self.beam_width]

    def predict(self, X):
        out, probs = self.sess.run([self.sy_predictions, self.sy_probs],
            feed_dict= {
                self.sy_X: X
            })

        return out, probs

