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


from mnist_reader import data_set, mnist_data_set


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
# def perform_DoG(incoming,W1,W2):
#
#     input = T.tensor4('inputs')
#     conv = lasagne.layers.InputLayer(
#             shape=(None, 1, 28, 28),
#             input_var=input)
#     conv=std_conv_layer(conv,1,7,W=T.concatenate([W1,W2],axis=1)) # convi=uling wih two filters with different sigma
#     conv_output=lasagne.layers.get_output(conv)#this will be ,2,h,w tensor
#     conv_function=theano.function([input],conv_output)
#
#     conv_ouput_actual=conv_function(incoming)
#     # conv_output_1=conv_output_actual_actual[:,0,:,:]
#     # conv_output_2=conv_output_actual[:,1,:,:]
#
#     return conv_output_actual

###############################################################################
# W1=sio.loadmat('W1.mat')['h']
# W2=sio.loadmat('W2.mat')['h']
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
def plot_weights(W, plot_name, file_path = '.',
                    max_subplots = 100, max_figures = 32,
                    figsize = (28, 28)):
    W = W.get_value(borrow = True)
    W=np.reshape(W,(2,28,28,1024))
    W=np.swapaxes(W,0,3)
    W=np.swapaxes(W,3,1)
    W=np.swapaxes(W,2,3)
    shape = W.shape
    assert((len(shape) == 2) or (len(shape) == 4))
    max_val = np.max(W)
    min_val = np.min(W)

    if len(shape) == 2:
        plt.figure(figsize = figsize)
        plt.imshow(W, cmap = 'gray',#'jet',
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
        figs, axes = plt.subplots(nrows, ncols, figsize = figsize,
                                  squeeze = False)
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        for i, (r, c) in enumerate(iter_product(range(nrows), range(ncols))):
            if i >= shape[1]:
                break
            im = axes[r, c].imshow(W[j, i], cmap = 'gray',#'jet',
                                   vmax = max_val, vmin = min_val,
                                   interpolation = 'none')
        figs.colorbar(im, ax = axes.ravel().tolist())
        file_name = plot_name + '_fmap' + str(j) + '.png'
        plt.savefig(os.path.join(file_path, file_name))
        plt.clf()
        plt.close()
    return

###############################################################################
class DoG_Layer(layers.Layer):

    def __init__(self,incoming,**kwargs):

        super(DoG_Layer, self).__init__(incoming, **kwargs)
        # global W1,W2
        _,self.channels,self.height,self.width=incoming.shape

        # W1 = T.reshape(W1,[1,1,height,width])
        # W2=T.reshape(W2,[1,1,height,width])
        # self.W=T.concatenate([W1,W2],axis=1)
        self.time_steps=32


    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[0] = self.time_steps
        output_shape[1] = output_shape[1]*2 # two maps are created one for ON centered and OFF centered DoG
        return tuple(output_shape)

    def get_output_for(self, incoming, **kwargs):
        conv_output = T.nnet.conv2d(incoming, dog_W, filter_flip=False,
                                    border_mode='half', subsample=(1, 1))

        #conv_output=dog_conv(incoming) #expected dimension:1,2,h,w

        conv_output2=conv_output[:,::-1,:,:]

        dog_maps=conv_output - conv_output2

        dog_maps=T.ge(dog_maps,0)*dog_maps
        dog_maps = T.switch(dog_maps>0.0, dog_maps, 0.0)
        dog_maps_neq_Zero=T.neq(T.reshape(dog_maps,(self.height*self.width*self.channels*2,)),0)

        dog_maps=T.floor(dog_maps*self.time_steps)
        dog_maps=T.switch(dog_maps<self.time_steps-1,dog_maps,self.time_steps-1)
        dog_maps=T.cast(dog_maps,'int32')
        dog_maps = self.time_steps-1-dog_maps
        dog_maps = T.reshape(dog_maps,(self.height*self.width*self.channels*2,))




        output = T.zeros((self.time_steps,self.height*self.width*self.channels*2))
        #T.set_subtensor(output[dog_maps[:],T.arange(output.shape[1])]=1
        output = T.set_subtensor(output[dog_maps,T.arange(output.shape[1])],dog_maps_neq_Zero)
        output = T.reshape(output,[self.time_steps,self.channels*2,self.height,self.width])
        #print(dog_maps[:].eval())
        return output[-1:]

####################################################################################
class snn_denseLayer(DenseLayer):


        def __init__(self,incoming,num_units,batch_size,snn_enabled=True,threshold=64,refractory_voltage=-10000,**kwargs):

            #super(DoG_Layer, self).__init__(incoming, **kwargs)
            #self.v_in=T.fvector()

            # self.v=v.astype(theano.config.floatX)
            # _,self.channels,self.height,self.width=incoming.shape
            self.incoming=incoming

            # self.batch_size=incoming.input_var.shape[0]
            self.batch_size=batch_size
            self.snn_enabled=snn_enabled
            self.threshold=threshold
            self.refractory_voltage=refractory_voltage
            #self.output_spike=T.zeros([1,num_units])
            super(snn_denseLayer, self).__init__(incoming, num_units, W=lasagne.init.Normal(std=0.01, mean=0.8), **kwargs)

        def get_output_for(self, input, deterministic=False, **kwargs):

            # print(super(snn_denseLayer, self).get_output_for(input, **kwargs))
            self.input=input
            v=self.v_in+super(snn_denseLayer, self).get_output_for(input, **kwargs)
            # v=super(snn_denseLayer, self).get_output_for(input, **kwargs)
            vmax=T.max(v)
            flag=T.gt(vmax,self.threshold)
            self.output_spike=T.switch(T.eq(vmax,v),flag,0.0)
            self.v_out=flag*self.refractory_voltage + (1.0-flag)*v
            #sample_net.do_stdp()
            return self.output_spike

        def set_inputs(self,V,H_in):
            self.v_in=V
            self.H_in=H_in

        def get_output_shape(self):
            return (self.batch_size,self.num_units),self.incoming.shape,(self.batch_size,self.num_units)

        def do_stdp(self):
            # self.H_in = T.tensor4()
            self.H_out=self.H_in+self.input
            # W = self.W*(1-self.W)
            # # self.update=W
            # # W=T.reshape(W,(1,T.shape(W)[0],T.shape(W)[1]))
            # # W=T.repeat(W,self.batch_size,0)
            # W=T.tile(W,(self.batch_size,1,1))
            # output_spike=self.output_spike
            # # # output_spike=T.reshape(output_spike,(T.shape(output_spike)[0],1,T.shape(output_spike)[1]))
            # output_spike=output_spike.dimshuffle(0,'x',1)
            # # #W=T.addbroadcast(W,1)
            # # #output_spike=output_spike.T
            # output_spike=T.addbroadcast(output_spike,1)
            # #
            # # #dim0,dim1,dim2,dim3=T.shape(self.H_out)
            # #
            # H_out_reshaped=T.reshape(self.H_out,(T.shape(self.H_out)[0],-1,1))
            # #
            # #
            # H_out_reshaped=T.addbroadcast(H_out_reshaped,2)
            W=self.W #Dx1024
            W=W.dimshuffle('x',0,1)#1XDX1024
            W=T.addbroadcast(W,0)
            output_spike=self.output_spike
            output_spike=output_spike.dimshuffle(0,'x',1)
            output_spike=T.addbroadcast(output_spike,1)
            H_out_reshaped=T.reshape(self.H_out,(T.shape(self.H_out)[0],-1,1))
            H_out_reshaped=T.addbroadcast(H_out_reshaped,2)


            update=((W*(1-W))*output_spike*T.switch(T.eq(H_out_reshaped,0.0),-1.0,1.0))
            sum_update=T.sum(update,axis=0)
            # # print('sum update : ')
            # # print(sum_update.type)
            count_nez=T.switch(T.eq(update,0.0),0.0,1.0)
            # # count_nez=T.neq(update,0.0)
            # # # print("TYpe : ***")
            # # # print(count_nez.dtype)
            count_update=T.sum(count_nez,axis=0)
            # # print(count_nez.type)#int8
            # # print(count_update.type)#int64
            count_update=T.switch(T.eq(count_update,0.0),1,count_update)
            count_update=T.cast(count_update,dtype=theano.config.floatX)
            # # # print('count_update')
            # # # print(count_update.dtype)
            update=sum_update/count_update
            # print(update.type)
            # update=T.cast(update,dtype=theano.config.floatX)
            self.update=update
            #W+=self.update
            return self.update,self.H_out
###################################################################################

class snn():
    def __init__(self, input_shape, input = None,
                 num_class = 10,
                 name = 'snn_autonet'):
        if input is None:
            input = T.tensor4()#
        self.input_shape=input_shape
        self.input = input
        self.name = name
        self.num_class = num_class
        self.time_steps=32
        self.batch_size=self.input_shape[0]

        print 'computing DoG maps ...'

        self.DoG_maps=self.dog_output(input) # the input passed to the class is simply the image, the DoG maps are calculated using
        #this image. A slice of this DoG map  is given as the input to the graph created
        # self.get_dog_map=theano.function(inputs=[input],outputs=self.DoG_maps)


        self.input_shape[1]=self.input_shape[1]*2

        #input_layer = InputLayer(shape = input_shape, input_var = T.reshape(self.DoG_maps[0],[1,self.channels,self.height,self.width]))


        self.layers =[
                    ('snn_dense1',snn_denseLayer , [1024,self.batch_size])  #num_units is initialised as 10

                ]
        #self.layer_names = ['input'] + [attr[0] for attr in layers]

        #self.all_layers, _ = self.create_net(layers, input_layer)
        self.create_snn()


    def dog_output(self,input_image):

        _,self.channels,self.height,self.width=input_image.shape
        conv_output = T.nnet.conv2d(input_image, dog_W, filter_flip=False,
                                    border_mode='half', subsample=(1, 1))


        conv_output2=conv_output[:,::-1,:,:]

        dog_maps=conv_output - conv_output2
        #
        dog_maps=T.ge(dog_maps,0)*dog_maps
        dog_maps = T.switch(dog_maps>0.0, dog_maps, 0.0)
        dog_maps_neq_Zero=T.neq(T.reshape(dog_maps,(self.batch_size*self.height*self.width*self.channels*2,)),0)
        #
        dog_maps=T.floor(dog_maps*self.time_steps)
        dog_maps=T.switch(dog_maps<self.time_steps-1,dog_maps,self.time_steps-1)
        dog_maps=T.cast(dog_maps,'int32')
        dog_maps = self.time_steps-1-dog_maps
        dog_maps = T.reshape(dog_maps,(self.batch_size*self.height*self.width*self.channels*2,))

        output = T.zeros((self.time_steps,self.batch_size*self.height*self.width*self.channels*2))
        #
        output = T.set_subtensor(output[dog_maps,T.arange(output.shape[1])],dog_maps_neq_Zero)
        output = T.reshape(output,[self.time_steps,self.batch_size,self.channels*2,self.height,self.width])
        # #print(dog_maps[:].eval())
        return output

    def plot_weights(self, file_path = '.', plot_id = 0,
                     max_subplots = 64, max_figures = 16, layer_id = -1,
                     figsize = (6, 6)):
        i = -1
        # for name, layer in zip(self.layer_names, self.layer_list):
        #     try:
        #         # W = T.reshape(layer.W,(1,2,28,28))
        #         #W = layer.W
        #     except:
        #         continue
        #     i += 1
        #     if layer_id > -1 and i != layer_id:
        #         continue
        W=self.all_layers[-1].W
        #W = T.reshape(W,(1,2,28,28))
        plot_weights(W, 'ID' + str(plot_id) + '_' + 'snn', file_path,
                            max_subplots, max_figures, figsize)
    def create_net(self, layer_details, prev_layer):
        print self.name, '.create_net> building net...'
        layers_list = [prev_layer]
        layers_dict = {'input': prev_layer}
        for attributes in layer_details:
            name, layer_fn = attributes[: 2]
            params = []
            params_dict = {}
            if len(attributes) >= 4:
                params, params_dict = attributes[2: 4]
            elif len(attributes) >= 3:
                params = attributes[2]
            print 'layer: ', name
            prev_layer = layer_fn(prev_layer, *params, **params_dict)
            layers_dict[name] = prev_layer
            layers_list.append(prev_layer)
        print 'done.'
        return layers_list, layers_dict




    def create_snn(self,layers='None'):
        print 'Building snn...'
        if(layers=='None'):
            layers=self.layers
        input_layer = InputLayer(shape = self.input_shape, input_var =  T.reshape(self.DoG_maps[0],self.input_shape)) #the input layer of
        #the graph which takes a slice of DoG map.
        all_layers, _ = self.create_net(layers, input_layer)
        self.all_layers=all_layers
        LR=T.scalar()

        def fn(*args):
            #args[0] - input slice of DoG map
            #args[1] - output_spike train
            #args[2] - v_in for snn DenseLayer
            #args[3] - h_in for snn DenseLayer ** not present if snn enabled is false
            #.
            #.
            #.
            args=list(args)
            print(args)
            # for i in range(1,len(args)-1):
            #     args[i]=args[i][0]

            print(len(args))
            print('args')
            print(args)
            i=2
            for layer in (all_layers[1:]):

                if(layer.snn_enabled):
                    layer.v_in = args[i]
                    layer.H_in = args[i+1]
                    i+=2
                else:
                    layer.v_in=args[i]
                    i+=1

            all_layers[0].input_var=args[0]
            # #all_layers[0].input_var=T.reshape(args[0],(1,2,28,28))
            output_spike_train=lasagne.layers.get_output(all_layers[-1]) #the graph is created
            # print(T.shape(output_spike_train))
            vH_out_list=[]
            #H_out_list=[]
            W_dict=[]
            #
            for layer in all_layers[1:]:
                vH_out_list.append(layer.v_out)
                if (layer.snn_enabled):
                    layer.do_stdp()
                    vH_out_list.append(layer.H_out)
                    W_dict.append((layer.W,layer.W+LR*layer.update))
            print('fn returning : ')
            # print([output_spike_train]+vH_out_list)

            return [output_spike_train]+vH_out_list, W_dict
            #return vH_out_list

        def set_outputs_info():
            output=[]

            #initial_spike_train=T.zeros(all_layers[-1].get_output_shape()[2])
            initial_spike_train=T.zeros((self.batch_size,self.all_layers[-1].num_units))
            print(T.shape(initial_spike_train))

            #output.append(initial_spike_train)

            vH_list=[]

            # for layer in all_layers[1:]:
            #     layer.set_inputs(T.vector(),T.tensor4())

            for layer in all_layers[1:]:
                # print(T.zeros(layer.get_output_shape()[0])
                vH_list.append(T.zeros(layer.get_output_shape()[0]))
                if (layer.snn_enabled) :
                # print()
                    vH_list.append(T.zeros(layer.get_output_shape()[1]))

            output=[initial_spike_train]+vH_list
            #output=vH_list

            print(output)
            #output = [T.shape_padleft(a) for a in output]
            # for i,a in enumerate(output):
            #     output[i]=T.shape_padleft(a)

            print('set output info :')
            print(output)
            #print(T.shape(output))
            return output

        # theano.printing.pydotprint(self.DoG_maps, outfile="./debug.png", var_with_name_simple=True)
        components,updates = theano.scan(fn,sequences=[self.DoG_maps], non_sequences=LR,outputs_info=set_outputs_info())

        #print(T.shape(components))
        shape=T.shape(components[0])
        output=T.sum(components[0],axis=0)
        output=T.switch(T.ge(output,1.0),1.0,output)
        output=T.cast(output,dtype=theano.config.floatX)#128x1024
        time_peaked=T.sum(components[0],axis=2) #32x128
        real_valued=T.argmax(time_peaked,axis=0)
        real_valued=(32-real_valued)/32.0
        factor=T.sum(time_peaked,axis=0)#to take care of no spike
        factor=factor*real_valued#128,
        factor=T.reshape(factor,[T.shape(factor)[0],1])
        factor=T.addbroadcast(factor,1)
        output=output*factor

	delta_weight=T.zeros((1))
        print('*********')
        print(delta_weight)

        for key,value in updates.iteritems():
            delta_weight+=T.mean(abs(value-key))

        delta_weight/=len(updates.keys())



        self.train=theano.function(inputs=[self.input,LR],outputs=[components[0],delta_weight],updates=updates,on_unused_input='ignore')

        self.test=theano.function(inputs=[self.input,LR],outputs=output)


        print('compiled')

    def test_batch(self,X):
        return self.test(X.astype(theano.config.floatX),0.0).astype(theano.config.floatX)
        #self.get_weights=theano.function([],all_layers[-1].W)

    # def test_batch(self,X):
    #     classifier_input=np.zeros((np.shape(X)[0],1024),dtype=theano.config.floatX)
    #     for i in range(0,np.shape(X)[0]):
    #         slice_output=self.test(X[i:i+1],0)
    #         classifier_input[i]=slice_output
    #
    #     return classifier_input

class count_classifier():
    def __init__(self,name='count_classifer',num_classes=10,num_units=1024):
        # self.snn_network=snn_network
        self.name=name
        # self.input_shape=input_shape
        shape=(num_classes,num_units)
        self.histogram=np.zeros(shape)

    def train(self,X,Y,p=1):
	#self.sumx=np.sum(X)
        i=0
	while(i<Y.size):
		self.histogram[Y[i:i+1],:]+=X[i:i+1]
		i+=1
	#self.sumh=np.sum(self.histogram)
	if (p==0):
        	return self.test(X,Y)

    def test(self,X,Y):
        neuron_spiked=np.argmax(X,axis=1)
        prediction=np.argmax(self.histogram[:,neuron_spiked],axis=0)
        output=(prediction==Y)
        accuracy=np.sum(output)/128.0
	self.output=output
	self.neuron_spiked=neuron_spiked
	self.prediction=prediction
        return accuracy
class softmax_classifier():
    def __init__(self,name='softmax_classifer'):
        # self.snn_network=snn_network
        self.name=name
        # self.input_shape=input_shape
        self.build_classifer()

    def build_classifer(self):
        print('Building classifier...')
        self.input = T.matrix('inputs')
        momentum=0.9
        # classifier_input=self.snn_network.test_batch(input)
        target = T.ivector('targets')
        LR = T.scalar('LR', dtype=theano.config.floatX)

        input_layer=lasagne.layers.InputLayer(
            shape=(None,1024),input_var=self.input)

        # dense_layer=lasagne.layers.DenseLayer(input_layer,num_units=128,
        #                                 W=lasagne.init.HeNormal(1.0),
        #                                 nonlinearity=lasagne.nonlinearities.rectify)

        self.output_layer=lasagne.layers.DenseLayer(input_layer,num_units=10,
                                        W=lasagne.init.HeNormal(1.0),
                                        nonlinearity=lasagne.nonlinearities.softmax)


        print('Done')
        print('Building theano functions...')

        self.params = lasagne.layers.get_all_params(self.output_layer,
                                                           trainable = True)

        self.train_output = lasagne.layers.get_output(self.output_layer)
        self.test_output = self.train_output

        Y = T.ivector()
        train_error = lasagne.objectives.\
                        categorical_crossentropy(self.train_output, Y).mean()
        train_loss = train_error
        train_accuracy = T.mean(T.eq(T.argmax(self.train_output, axis = 1), Y),
                                dtype = theano.config.floatX)
        test_loss = lasagne.objectives.\
                        categorical_crossentropy(self.test_output, Y).mean()
        test_accuracy = T.mean(T.eq(T.argmax(self.test_output, axis = 1), Y),
                                dtype = theano.config.floatX)

        self.test_func = theano.function(
                                inputs = [self.input, Y],
                                outputs = [test_loss, test_accuracy,
                                           self.test_output])

        LR= T.scalar()
        updates = lasagne.updates.momentum(train_loss, self.params,
                                           learning_rate = LR,
                                           momentum = momentum)
        self.train_func = theano.function(
                                inputs = [self.input, Y, LR],
                                outputs = [train_loss, train_accuracy],
                                updates = updates)

        print 'Done.'

    def test(self, X, Y):
        loss, accuracy, confidences = self.test_func(X, Y)
        # print(np.shape(self.snn_network.test_batch(X)))
        return confidences, loss, accuracy

    def train(self, X, Y, LR):
        loss, accuracy = self.train_func(X, Y, LR)
        return loss, accuracy
        # print(self.snn_network.test_batch(X))












####################################################################################


if __name__ =='__main__' :
    np.random.seed(11)
    data_path = '/data3/deepak_interns/vikram/vikram/mnist/'
    batch_size = 128

    input = T.tensor4('inputs')
#
# #     # W1 = T.matrix('W1')
# #     # W2 = T.matrix('W2')
    sample_net = lasagne.layers.InputLayer(
            shape=(None, 2, 28, 28),
            input_var=input)
# #     sample_net=DoG_Layer(sample_net)
# #
# #     sample_net2 = lasagne.layers.InputLayer(
# #             shape=(None, 1, 28, 28),
# #             input_var=input)
# #     sample_net2=DoG_Layer(sample_net2)
# #
# #
# #
# #     dog_layer_output=lasagne.layers.get_output(sample_net2)
# #     snn_dog_function=theano.function([input],dog_layer_output)
# #     # dog_output=conv_output=lasagne.layers.get_output(sample_net)
# #     # dog_function=theano.function([input],dog_output)
# #
    # sample_net=snn_denseLayer(sample_net,num_units=10,threshold=64,refractory_voltage=-100,batch_size=self.batch_size)
    # sample_net.set_inputs(T.matrix(),T.tensor4())
# #
# # #     print(sample_net.v_in,sample_net.H_in)
#     snn_denseLayer_output=lasagne.layers.get_output(sample_net)
#     sample_net.do_stdp()
#     snn_function=theano.function([input,sample_net.v_in],[snn_denseLayer_output,sample_net.v_out],on_unused_input='ignore')
# # #
#     # sample_net.do_stdp()
#     snn_function_print=theano.function([input,sample_net.v_in,sample_net.H_in],[sample_net.output_spike,sample_net.H_out,sample_net.H_in,sample_net.update])
    # snn_function_output_spike=theano.function([input,sample_net.v_in],[sample_net.output_spike])
# #
    print 'Loading Dataset'
    mnist = mnist_data_set(data_path, batch_size)
    print 'done loading'
    datasets = mnist.data_sets
    print type(datasets['train'].X)
    print np.max(datasets['train'].X), np.min(datasets['train'].X)
    print datasets['train'].X.shape, datasets['train'].Y.shape
    print datasets['test'].X.shape, datasets['test'].Y.shape
    print datasets['valid'].X.shape, datasets['valid'].Y.shape
    #mnist.visualize_mnist(datasets['train'].X,datasets['train'].Y)

    print(datasets['train'].X.shape)
    for i, (X, Y) in enumerate(datasets['train']):
        sio.savemat('input_image1_batch128',{'img':X[100]})
        print(X.shape)
        break
    print(X.shape)
    x=X
# #
# #     print('output from DoG layer')
# #     dog_ans=snn_dog_function(x)
# #     print(dog_ans)
# #     print(np.sum(dog_ans))
# #     # dict={'w1' : x[0,0,:,:]}
# #     # W1=sio.loadmat('W1.mat')
# #     # W2=sio.loadmat('W2.mat')
# #
# #     # ans=dog_function(x)
    # ans,v=snn_function(x,np.zeros(10).astype(theano.config.floatX)+30)
    # print(ans.shape)
    # print(v.shape)
# #
# #     # print(ans[:,0,:,:].shape)
# #     # dict2={'on' : ans[:,0,:,:],'off' : ans[:,1,:,:] }
# #     # print(ans)
# #     #print(x)
    # print(ans)
# #     print(np.sum(ans))
    # print(v)
# #
# #     print("all values \n \n")
# #
# #     all_values = snn_function_print(x , np.zeros(10).astype(theano.config.floatX)+30 , np.zeros([1,2,28,28]).astype(theano.config.floatX))
# #     for print_value in all_values:
# #         print(print_value)
# #
# #
# #     print(np.sum(dog_ans-all_values[1]))
# #
# #     print(all_values[-1].shape)
# #
# #     sio.savemat('testing_stdp',{'H_out' : all_values[1], 'updates' : np.reshape(all_values[-1],(2,28,28,10)) })
# #     #print(np.sum(np.sign(all_values[3]+np.abs(all_values[3]))-all_values[1]))
# #
# #     # print(T.reshape(all_values[1]))
# #
# #     # print(np.max(ans))
# #     # print(np.min(ans))
# #     # print(x)
# #     # plt.imshow(ans[0,1,:,:],cmap='gray')
# #     # plt.show()
# #     # plt.savefig('off.png')
# #     # print(W1['h'].shape)
# #     # sio.savemat('w1',dict)
# #     # sio.savemat('output',dict2)
# #     ####################################################################################################
#     #testing snn class
    # print(x.shape)

    snn_object=snn([16,1,28,28])
    print(datetime.datetime.now())
    # print(snn_object.all[layers[-1]].batch_size)
    components=snn_object.train(X[0:16],1.0)
    # output=snn_object.test(X,0)
    # print(output.shape)
    print(datetime.datetime.now())
    # for a in components:
    #     print(a)
    #     print(a.shape)
    sio.savemat('batchwise',{'output_spike':components[0],'v_out':components[1]})

    # dog=snn_object.get_dog_map(X)
    # print(dog[0].shape)







    # print(anss.shape)
    # print(np.sum(ans[:,100,:,:,:]))
    # sio.savemat('batchwise_input1',{'xtest1':ans[:,100,:,:,:]})
    # print(snn_object.DoG_maps.eval().shape)

    # ans=snn_function(dog[27],np.zeros([128,10],dtype=theano.config.floatX)+30)
    # # dog2=sio.loadmat('input1.mat')['xact1']
    # # print(dog2.shape)
    # # print(np.reshape(dog2[0],[1,2,28,28]))
    # # ans=snn_function(np.reshape(dog2[0],[1,2,28,28]))
    # print(ans[0].shape,ans[1].shape)
    #
    # # print(sample_net.W.eval().shape)
    #
    # for a in snn_function_print(dog[27],np.zeros([128,10],dtype=theano.config.floatX)+30,dog[26]):
    #     print(a)
    #     print(a.shape)
    # print(v.shape)
# #     w1=np.array(snn_object.all_layers[-1].W.eval())
# #     print(w1)
#     components=snn_object.train(x,0.01)
#
#
#     model_save_path = './models'
#     path = os.path.join(model_save_path, 'train1')
#     if not os.path.exists(path):
#         os.makedirs(path)
#         os.makedirs(os.path.join(path, 'snapshots'))
#     # train1_net(network, datasets, path)
#     print('saving the trained spiking network...')
#     f = open(os.path.join(path, snn_object.name + '.save'), 'wb')
#     sys.setrecursionlimit(50000)
#     cPickle.dump(snn_object, f, protocol = cPickle.HIGHEST_PROTOCOL)
#     f.close()
#     print('Done')
#
#     print 'loading trained snn for training classifier...'
#     f = open(os.path.join(path, snn_object.name + '.save'), 'rb')
#     snn_loaded_object=cPickle.load(f)
#     f.close()
#     print('Done')
#
#     output=snn_loaded_object.test(x,0)
#     np.set_printoptions(threshold=np.inf)
#     print(output)
#
#
#     # sum1=T.argmax(T.sum(T.squeeze(components[0]),0)).eval()
#     # print(sum1)
#     # sum2=T.argmax(T.sum(components[0],1)).eval()
#     # sum3=T.argmax(T.sum(components[0],2)).eval()
#     # print(sum1,sum2,sum3)
#     # x=np.squeeze(components[0])
#     # print(x)
#     # sum1=
#     # shape=T.shape(components[0])
#     # # print(shape[0])
#     # x=T.reshape(components[0],(shape[0],shape[2]))
#     # # print(T.shape(x).eval())
#     # # print(T.sum(x,0).eval())
#     # time_slot_peaked=T.argmax(T.sum(x,1))
#     # factor=(32 - time_slot_peaked)/32.0
#     # output=T.sum(x,0)
#     # output=output*factor
#     # np.set_printoptions(threshold=np.inf)
#     # print(np.sum(output.eval()))
#     # print(T.argmax(T.sum(x,1)).eval())
# #     w2=np.array(snn_object.all_layers[-1].W.eval())
# #
# #     print(w2-w1)
# #
# #     #print(np.shape((w2-w1).eval()))
# #     #print(T.shape(components))
# #     print(components)
# #
# #     for a in components:
# #         print(np.shape(a))
# #     #
# #     # sio.savemat('stdp_output',{'output_spike':components[0],'v_out':components[1],'h_out':components[2]})
# #     # sio.savemat('dog_output',{'dog_output':snn_object.dog_output(x).eval()})
# #     sio.savemat('weight_update32',{'updates':np.reshape((w2-w1),(2,28,28,10))})
# #
# #     #print(T.argmax(T.argmax(components[0],2)).eval(),T.argmax(T.argmax(components[0],0)).eval())
# #     x=np.argmax(np.squeeze(components[0]))
# #     print('spiked in time step : %d , neuron : %d'%(np.floor_divide(x,10),np.mod(x,10)))
# #     #print()
