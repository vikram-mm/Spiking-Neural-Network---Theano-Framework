'''
Created on 25-May-2017

@author: vikram
'''


import numpy as np
# import cv2
import os
import random
import scipy.io
import copy
import theano
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cPickle as pickle
from numpy import dtype

import gzip


class data_set():


    def __init__(self, X, Y, batch_size = 1, do_shuffle = False):
        random.seed(11)
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle
        self.num_samples = X.shape[0]
        assert(self.num_samples == Y.shape[0])
        self.data_item_dimension = X.shape[1: ]


    #def __iter__(self):
    #    return self


    #? Last batch incomplete batch is DROPPED.
    def __iter__(self):
        if self.do_shuffle:
            #print 'SHUFFLED...'
            ##??? FULL DATA NOT USED. UNCOMMENT.
            indices = np.random.permutation(self.num_samples)
            #indices = np.random.permutation(20)
            #print indices
        else:
            indices = np.arange(self.num_samples)

        ##??? FULL DATA NOT USED. UNCOMMENT.
        for i in range(0, self.num_samples, self.batch_size):
        #for i in range(0, 20, self.batch_size):
            if i + self.batch_size > self.num_samples:
                break
            X_orig = self.X[indices[i: i + self.batch_size]]
            X_aug = X_orig

            # if self.do_shuffle:
            #     X_aug = np.empty(X_orig.shape, dtype = np.float32)
            #     flip_flags = np.random.randint(2, size = X_orig.shape[0])
            #     padded = np.pad(X_orig, ((0,0), (0,0), (4,4), (4,4)), mode='constant')
            #     random_cropped = np.zeros(X_orig.shape, dtype = np.float32)
            #     crops = np.random.random_integers(0, high = 8, size = (X_orig.shape[0], 2))
            #     for r, flip in enumerate(flip_flags):
            #         tmp = padded[r, :, crops[r, 0]: (crops[r, 0] + 32), crops[r, 1]: (crops[r, 1] + 32)]
            #         if flip == 1:
            #             tmp = tmp[:, :, :: -1]
            #         X_aug[r, :, :, :] = tmp

            yield (X_aug, self.Y[indices[i: i + self.batch_size]])



class mnist_data_set():


    def __init__(self, data_path, batch_size = 1,
                 valid_set_size_percent = 0.1):
        random.seed(11)
        self.data_path = data_path
        X_train, Y_train, X_test, Y_test = self.load_mnist()
        #self.X_mean = np.mean(X_train, axis = 0)
        valid_set_size = np.ceil(X_train.shape[0] * valid_set_size_percent)
        X_train, Y_train, X_valid, Y_valid = \
            self.create_validation_set(X_train, Y_train, valid_set_size)
        # self.X_mean = np.mean(X_train, axis = 0)
        # X_train -= self.X_mean
        # X_valid -= self.X_mean
        # X_test -= self.X_mean
        # print 'Mean subtracted dataset.'
        self.data_sets = {
                        'train': data_set(X_train, Y_train, batch_size, True),
                        'test': data_set(X_test, Y_test, batch_size),
                        'valid': data_set(X_valid, Y_valid, batch_size)}



    def load_mnist_images(self,filename):
	with gzip.open(filename, 'rb') as f:
	    data = np.frombuffer(f.read(), np.uint8, offset=16)
	data = data.reshape(-1, 1, 28, 28)
	# The inputs come as bytes, we convert them to float32 in range [0,1].
	return data / np.float32(255.0)

    def load_mnist_labels(self,filename):
	with gzip.open(filename, 'rb') as f:
	    data = np.frombuffer(f.read(), np.uint8, offset=8)
	return data


    def load_mnist(self):
	X_train = self.load_mnist_images(self.data_path+'train-images-idx3-ubyte.gz')
	Y_train = self.load_mnist_labels(self.data_path+'train-labels-idx1-ubyte.gz')
	X_test = self.load_mnist_images(self.data_path+'t10k-images-idx3-ubyte.gz')
	Y_test = self.load_mnist_labels(self.data_path+'t10k-labels-idx1-ubyte.gz')
	return X_train, Y_train, X_test, Y_test


    def create_validation_set(self, X, Y, valid_set_size):
        return (X[valid_set_size: ], Y[valid_set_size: ],
                X[: valid_set_size], Y[: valid_set_size])

    def visualize_mnist(self, X_train, Y_train, samples_per_class = 10):
        class_names = ['0', '1', '2', '3', '4',
                       '5', '6', '7', '8', '9']
        num_classes = len(class_names)

        for y, cls in enumerate(class_names):
            idxs = np.flatnonzero(Y_train == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                X = X_train[idx]
                print(X.shape)
                plt.imshow(X[0,:,:], cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title(cls)

        plt.show()
        plt.savefig('trial.png')
if __name__ =='__main__' :
    data_path = '/data3/deepak_interns/vikram/vikram/mnist/'
    batch_size = 128

    print 'Loading Dataset'
    mnist = mnist_data_set(data_path, batch_size)
    print 'done loading'
    datasets = mnist.data_sets
    print type(datasets['train'].X)
    print np.max(datasets['train'].X), np.min(datasets['train'].X)
    print datasets['train'].X.shape, datasets['train'].Y.shape
    print datasets['test'].X.shape, datasets['test'].Y.shape
    print datasets['valid'].X.shape, datasets['valid'].Y.shape
    mnist.visualize_mnist(datasets['train'].X,datasets['train'].Y)
