import numpy as np
import theano
import theano.tensor as T
import timeit
#import pickle
import cPickle
import os
import datetime
# import cv2
import lasagne
import random
import matplotlib
from numpy import dtype
from collections import OrderedDict
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from theano.compile.nanguardmode import NanGuardMode


from calface_reader import data_set, mnist_data_set


from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import GlobalPoolLayer as GapLayer
from lasagne.nonlinearities import softmax, sigmoid
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import batch_norm
from lasagne.nonlinearities import rectify
import scipy.io as sio

def relu1(x):
    return T.switch(x < 0, 0, x)

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from stdp import stdpOp


#from visualize import plot_conv_weights,

def std_conv_layer(input, num_filters, filter_shape, pad = 'same',
                   nonlinearity = lasagne.nonlinearities.rectify,
                   W = None,
                   #W = lasagne.init.Normal(std = 0.01, mean = 0.0),
                   b = lasagne.init.Constant(0.),
                   do_batch_norm = False):
    if W == None:
        if nonlinearity == lasagne.nonlinearities.rectify:
            print 'convlayer: rectifier func'
            W = lasagne.init.HeNormal(gain = 'relu')
        else:
            print 'convlayer: sigmoid func'
            W = lasagne.init.HeNormal(1.0)
    else:
        print 'convlayer: W not None'
    conv_layer = ConvLayer(input, num_filters, filter_shape,
                        pad = pad, flip_filters = False,
                        W = W, b = b,
                        nonlinearity = nonlinearity)
    if do_batch_norm:
        conv_layer = lasagne.layers.batch_norm(conv_layer)
    else:
        print 'convlayer: No batch norm.'
    return conv_layer

from snn_conv import snn
from lasagne import layers

try:
    from lasagne.layers import TransposedConv2DLayer as DeconvLayer
except:
    from new_conv import TransposedConv2DLayer as DeconvLayer


try:
    from lasagne.layers import ExpressionLayer
except:
    from new_special import ExpressionLayer


from itertools import product as iter_product


import sys

###############################################################################
W1=np.array([[  1.96519161e-05  , 2.39409349e-04,   1.07295826e-03,   1.76900911e-03,
    1.07295826e-03 ,  2.39409349e-04 ,  1.96519161e-05],
 [  2.39409349e-04  , 2.91660295e-03 ,  1.30713076e-02  , 2.15509428e-02,
    1.30713076e-02  , 2.91660295e-03 ,  2.39409349e-04],
 [  1.07295826e-03 ,  1.30713076e-02 ,  5.85815363e-02 ,  9.65846250e-02,
    5.85815363e-02 ,  1.30713076e-02 ,  1.07295826e-03],
 [  1.76900911e-03 ,  2.15509428e-02 ,  9.65846250e-02  , 1.59241126e-01,
    9.65846250e-02  , 2.15509428e-02 ,  1.76900911e-03],
 [  1.07295826e-03 ,  1.30713076e-02 ,  5.85815363e-02  , 9.65846250e-02,
    5.85815363e-02 ,  1.30713076e-02 ,  1.07295826e-03],
 [  2.39409349e-04 ,  2.91660295e-03 ,  1.30713076e-02  , 2.15509428e-02,
    1.30713076e-02 ,  2.91660295e-03 ,  2.39409349e-04],
 [  1.96519161e-05  , 2.39409349e-04 ,  1.07295826e-03  , 1.76900911e-03,
    1.07295826e-03  , 2.39409349e-04 ,  1.96519161e-05]])

W2=np.array([[ 0.00492233 , 0.00919613,  0.01338028 , 0.01516185 , 0.01338028 , 0.00919613,
   0.00492233],
 [ 0.00919613 , 0.01718062 , 0.02499766  ,0.02832606 , 0.02499766 , 0.01718062,
   0.00919613],
 [ 0.01338028 , 0.02499766 , 0.03637138 , 0.04121417 , 0.03637138 , 0.02499766,
   0.01338028],
 [ 0.01516185,  0.02832606 , 0.04121417 , 0.04670178,  0.04121417 , 0.02832606,
   0.01516185],
 [ 0.01338028 , 0.02499766 , 0.03637138 , 0.04121417 , 0.03637138,  0.02499766,
   0.01338028],
 [ 0.00919613 , 0.01718062 , 0.02499766 , 0.02832606 , 0.02499766 , 0.01718062,
   0.00919613],
 [ 0.00492233 , 0.00919613 , 0.01338028 , 0.01516185 , 0.01338028,  0.00919613,
   0.00492233]])
W=np.stack((W1,W2),axis=0)
dog_W=np.reshape(W,(2,1,7,7))

dog_W= dog_W.astype(theano.config.floatX)
###################################################################################

def plot_weights(W, plot_name, file_path = '.',color='Greens',
                    max_subplots = 100, max_figures = 32,
                    figsize = (28, 28)):
    try:
        #  W = W.get_value(borrow = True)
         W=W.eval()
    except:
        W=W
        # print('number of spikes')+str(np.sum(W))

    # W=np.reshape(W,(2,5,5,32))
    # W=np.swapaxes(W,0,3)
    # W=np.swapaxes(W,3,1)
    # W=np.swapaxes(W,2,3)
    shape = W.shape
    print(shape)
    assert((len(shape) == 2) or (len(shape) == 4))
    max_val = np.max(W)

    min_val = np.min(W)

    # print('max_val'+str(max_val)+'min_val'+str(min_val))

    if len(shape) == 2:
        plt.figure(figsize = figsize)
        plt.imshow(W, cmap = color,#'jet',
                   vmax = max_val, vmin = min_val,
                   interpolation = 'none')
        plt.axis('off')
        plt.colorbar()
        file_name = plot_name + '.png'
        plt.savefig(os.path.join(file_path, file_name))
        plt.clf()
        plt.close()
        return

    nrows = min(np.ceil(np.sqrt(shape[1])).astype(int),
                np.floor(np.sqrt(max_subplots)).astype(int))
    ncols = nrows
    '''
    max_val = -np.inf
    min_val = np.inf
    for i in range(shape[0]):
        tmp = np.mean(W[i], axis = 0)
        max_val = max(max_val, np.max(tmp))
        min_val = min(min_val, np.min(tmp))
    '''
    for j in range(min(shape[0], max_figures)):
        if(j==1):
            break
        figs, axes = plt.subplots(nrows, ncols, figsize = figsize,
                                  squeeze = False)
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        for i, (r, c) in enumerate(iter_product(range(nrows), range(ncols))):
            if i >= shape[1]:
                break
            im = axes[r, c].imshow(W[j, i], cmap = color,#'jet',
                                   vmax = 1.0, vmin =0.0,
                                   interpolation = 'none')
        figs.colorbar(im, ax = axes.ravel().tolist())
        file_name = plot_name
        plt.title('number of spikes : '+str(np.sum(W))+'\n'+'max : '+str(max_val)+' min :'+str(min_val))
        plt.savefig(os.path.join(file_path, file_name))
        plt.clf()
        plt.close()
    return
if __name__ =='__main__' :
	# def plot_weights(self, file_path = '.', plot_id = 0,
	#                  max_subplots = 64, max_figures = 16, layer_id = -1,
	#                  figsize = (6, 6)):
	#     i = -1
	#
	#     W=self.all_layers[-1].W
	#
	#     plot_weights(W, 'ID' + str(plot_id) + '_' + 'snn', file_path,
	#                         max_subplots, max_figures, figsize)

	np.random.seed(11)

	model_save_path = './models'

	path = os.path.join(model_save_path, 'train1')

	print 'loading snn'
	f = open(os.path.join(path, 'snn_autonet' + '.save'), 'rb')
	network=cPickle.load(f)
	f.close()
	print('Done')

	batch_size = 1

	#snn_loaded_object.all_layers[-1].stdp_enabled=False

	data_path = '/data3/deepak_interns/vikram/Face_Mbike/'

	print 'Loading Dataset'
	mnist = mnist_data_set(data_path, batch_size)
	print 'done loading'


	datasets = mnist.data_sets
	print type(datasets['train'].X)
	print np.max(datasets['train'].X), np.min(datasets['train'].X)
	
	
	data_shape = datasets['train'].X.shape
	data_shape = (batch_size, ) + data_shape[1: ]
	print 'Data shape:', data_shape

	def dog_output(input_image):

	    _,channels,height,width=input_image.shape
	    conv_output = T.nnet.conv2d(input_image, dog_W, filter_flip=False,
		                        border_mode='half', subsample=(1, 1))
	    time_steps=32

	    conv_output2=conv_output[:,::-1,:,:]

	    dog_maps=conv_output - conv_output2
	    #
	    dog_maps=T.ge(dog_maps,0)*dog_maps
	    dog_maps = T.switch(dog_maps>0.0, dog_maps, 0.0)
	    dog_maps_neq_Zero=T.neq(T.reshape(dog_maps,(batch_size*height*width*channels*2,)),0)
	    #
	    dog_maps=T.floor(dog_maps*time_steps)
	    dog_maps=T.switch(dog_maps<time_steps-1,dog_maps,time_steps-1)
	    dog_maps=T.cast(dog_maps,'int32')
	    # dog_maps = time_steps-1-dog_maps
	    # dog_maps = T.reshape(dog_maps,(batch_size*height*width*channels*2,))

	    # output = T.zeros((time_steps,batch_size*height*width*channels*2))
	    # #
	    # output = T.set_subtensor(output[dog_maps,T.arange(output.shape[1])],dog_maps_neq_Zero)
	    # output = T.reshape(output,[time_steps,batch_size,channels*2,height,width])
	    # #print(dog_maps[:].eval())
	    return dog_maps
	#new_snn=snn(data_shape)
	#colors=['Purples','Greens','Reds','YlOrBr']
	#new_snn.all_layers[-1].W=snn_loaded_object.all_layers[-1].W
	for i, (X) in enumerate(datasets['train']):
	    if(i==10):
		break
	    #dog_map=dog_output(X)
	    #plot_weights(dog_map,plot_name='dog'+str(i), file_path ='./models/plots/activation')
	    for j,test_func in enumerate(network.test_funcs):

		    output=test_func(X,0)
		    print 'output_shape : ',output.shape
		    #output=np.sum(output,axis=1,keepdims=True)

		    plot_weights(output,plot_name='output_spike_seperate'+str(i)+'_stage_'+str(j), file_path ='./models/plots/activation')
	
	

	#print(snn_loaded_object.all_layers[-1].W.get_value().shape)
