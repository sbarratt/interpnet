import pickle
import os
import glob
import numpy as np

for f in glob.glob('run*'):
    d = f
    net_params = pickle.load(open(os.path.join(d, 'net-params.pkl'), 'rb'))
    metrics = pickle.load(open(os.path.join(d, 'metrics.pkl'), 'rb'))
    try:
        print ("\n" + f)
        for param in ['embedding_size', 'num_hidden_lstm', 'captioning', 'dropout', 'num_hiddens']:
            print (param + ':', net_params.get(param))
        print ("classification epochs:", np.max(metrics['epoch_classifier']) + 1)
        print ("val_accuracy:", np.max(metrics['val_accuracy']))
        print ("explanation epochs:", np.max(metrics['epoch_explainer']) + 1)
        print ("val_loss_explanation:", metrics['val_loss_explanation'])
    except:
        pass
