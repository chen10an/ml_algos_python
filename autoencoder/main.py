# library
import numpy as np

# data
print("Importing data...")
from tensorflow.examples.tutorials.mnist import input_data

# files
import utils
from model import Autoencoder

# inputs
X_TRAIN = None
X_TEST = None

TEST = True

if __name__ == '__main__':
    # fill inputs
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X_TRAIN = mnist.train.images
    X_TEST = mnist.test.images

    print("\nX_TRAIN: {} with shape {}, dtype {}, and values in range {}".format(type(X_TRAIN),
    X_TRAIN.shape, X_TRAIN.dtype, (np.amin(X_TRAIN), np.amax(X_TRAIN))))
    print("X_TEST: {} with shape {}, dtype {}, and values in range {}".format(type(X_TEST),
    X_TEST.shape, X_TEST.dtype, (np.amin(X_TEST), np.amax(X_TEST))))

    print("\nShowing sample images from X_TRAIN... (close to continue)")
    utils.plotImgs(np.reshape(X_TRAIN, (X_TRAIN.shape[0],28,28)), 25, 5, cmap='gray_r')

    # model
    # default params are for mnist data
    model = Autoencoder(X_TRAIN, X_TEST, log_file='mnist_logs')
    model.train(model_file='mnist_model/model.ckpt')

    # plot
    print("\nTesting model on its input (train data)...")
    pred_train = model.sess.run(model.decoder, feed_dict={model.X: model.X_train})
    # compare
    print("Showing sample images from the model's reconstruction... (close to continue)")
    utils.plotImgs(np.reshape(pred_train, (pred_train.shape[0],28,28)), 25, 5, figsize=(5,5), cmap='gray_r')
    print("Showing sample images from the original data... (close to continue)")
    utils.plotImgs(np.reshape(model.X_train, (model.X_train.shape[0],28,28)), 25, 5, figsize=(5,5), cmap='gray_r')

    print("\nTesting model on test data...")
    pred_test = model.sess.run(model.decoder, feed_dict={model.X: model.X_test})
    # compare
    print("Showing sample images from the model's reconstruction... (close to continue)")
    utils.plotImgs(np.reshape(pred_test, (pred_test.shape[0],28,28)), 25, 5, figsize=(5,5), cmap='gray_r')
    print("Showing sample images from the original data... (close to finish)")
    utils.plotImgs(np.reshape(model.X_test, (model.X_test.shape[0],28,28)), 25, 5, figsize=(5,5), cmap='gray_r')
