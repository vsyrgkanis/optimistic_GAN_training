import h5py, numpy as np
from os.path import join

def load_data(mydir):
    train = h5py.File(join(mydir,'train.h5.batch1'))
    test = h5py.File(join(mydir,'valid.h5.batch1'))
    return ( (np.asarray(train['data']),np.squeeze(train['label'])), (np.asarray(test['data']),np.squeeze(test['label'])))

def sample_label(numclass, batch_size):
    label_idx = np.random.randint(0, numclass,size=(batch_size,2))
    label = np.zeros((batch_size,numclass))
    for idx,x in enumerate(label_idx):
        label[idx][x[0]] = 1.0
        label[idx][x[1]] = 1.0
    return label


def set_trainability(model, trainable=False):
	model.trainable = trainable
	for layer in model.layers:
		layer.trainable = trainable
