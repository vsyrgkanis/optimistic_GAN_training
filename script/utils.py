import h5py, numpy as np
from os.path import join
import keras.backend as K
from keras.layers.merge import _Merge

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

def modified_binary_crossentropy(target, output):
    return -1.0 * K.mean(target*output)

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((K.shape(inputs[0])[0], 1, 1, 1))
	return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty
