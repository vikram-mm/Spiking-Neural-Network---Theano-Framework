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
#from mnist_reader import data_set, mnist_data_set


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
def plot_weights(W, plot_name, file_path = '.',
                    max_subplots = 100, max_figures = 64,
                    figsize = (28, 28)):
    try:
   	W = W.get_value(borrow = True)
    except:
	W=W
	''' W=np.reshape(W,(2,5,5,32))
	W=np.swapaxes(W,0,3)
	W=np.swapaxes(W,3,1)
	W=np.swapaxes(W,2,3)'''
    #W = W/4
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
            im = axes[r, c].imshow(W[j, i], cmap = 'Greens',#'jet',
                                   vmax = max_val, vmin = min_val,
                                   interpolation = 'none')
        figs.colorbar(im, ax = axes.ravel().tolist())
        file_name = plot_name + '_fmap' + str(j) + '.png'
        plt.savefig(os.path.join(file_path, file_name))
        plt.clf()
        plt.close()
    return
###################################################################################
class snn_layer():
    '''
    Dummy class from which snn_denseLayer and snn_convLayer inherit

    '''



###################################################################################

class snn_denseLayer(DenseLayer,snn_layer):


        def __init__(self,incoming,num_units,batch_size,stdp_enabled=True,threshold=64,refractory_voltage=-10000,**kwargs):

            self.incoming=incoming
            self.batch_size=batch_size
            self.stdp_enabled=stdp_enabled
            self.threshold=threshold
            self.refractory_voltage=refractory_voltage
            super(snn_denseLayer, self).__init__(incoming, num_units, W=lasagne.init.Normal(std=0.01, mean=0.8), **kwargs)

        def get_output_for(self, input, deterministic=False, **kwargs):

            self.input=input
            v=self.v_in+super(snn_denseLayer, self).get_output_for(input, **kwargs)
            vmax=T.max(v)
            flag=T.gt(vmax,self.threshold)
            self.output_spike=T.switch(T.eq(vmax,v),flag,0.0)
            self.v_out=flag*self.refractory_voltage + (1.0-flag)*v
            return self.output_spike

        def set_inputs(self,V,H_in):
            self.v_in=V
            self.H_in=H_in

        def get_output_shape(self):
            return (self.batch_size,self.num_units),lasagne.layers.get_output_shape(self.incoming),(self.batch_size,self.num_units)

        def do_stdp(self):

            self.H_out=self.H_in+self.input
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
            count_nez=T.switch(T.eq(update,0.0),0.0,1.0)

            count_update=T.sum(count_nez,axis=0)

            count_update=T.switch(T.eq(count_update,0.0),1,count_update)
            count_update=T.cast(count_update,dtype=theano.config.floatX)

            update=sum_update/count_update

            self.update=update

            return self.update,self.H_out
###################################################################################

class snn_convLayer(ConvLayer,snn_layer):


        def __init__(self,incoming,stdp_enabled=False,threshold=10,
                                                        num_filters=4,filter_size=5,output_flag=1,refractory_voltage=-np.float32(10000000),**kwargs):

            #super(DoG_Layer, self).__init__(incoming, **kwargs)
            #self.v_in=T.fvector()

            # self.v=v.astype(theano.config.floatX)
            # _,self.channels,self.height,self.width=incoming.shape
            self.incoming=incoming
            self.num_filters=num_filters
            # self.batch_size=incoming.input_var.shape[0]
            #self.batch_size=batch_size
            self.stdp_enabled=stdp_enabled
	    self.output_flag=output_flag
            self.threshold=threshold
            self.refractory_voltage=refractory_voltage
            #self.output_spike=T.zeros([1,num_units])
            super(snn_convLayer,self).__init__(incoming, num_filters, filter_size,pad = 'same', flip_filters = False,
                         W = lasagne.init.Normal(std=0.05, mean=0.8), b = lasagne.init.Constant(0.),
                         nonlinearity = lasagne.nonlinearities.identity,**kwargs)

        def convolve(self, input, deterministic=False, **kwargs):

            # print(super(snn_denseLayer, self).get_output_for(input, **kwargs))
            self.input=input
            v=self.v_in+super(snn_convLayer, self).convolve(input,**kwargs)
            self.v_in=v
            # vmax=theano.tensor.signal.pool.pool_3d(v, ds=(3,3,self.num_filters), ignore_border=True,
            #                                     st=(1,1,1), padding=(1, 1, 0), mode='max',
            #                                       )

            shape=T.shape(v)
            vmax,arg_max=T.max_and_argmax(v,axis=1,keepdims=True)
            self.arg_max=arg_max
            #channelwise
            if self.stdp_enabled==False:

                tmp=T.switch(T.gt(vmax,self.threshold),1.0,0.0)
                output_spike=tmp*T.eq(T.arange(self.num_filters).dimshuffle('x',0,'x','x'),arg_max)
                v2=(1-tmp)*v+tmp*self.refractory_voltage*(1-output_spike)+tmp*self.refractory_voltage*output_spike
                self.output_spike=output_spike
                self.v_out=v2

                self.temp2=v2
                self.tempy=v2
                self.tempx=v2



            else:
		print('stdp enabled')
                temp2=theano.tensor.signal.pool.pool_2d(vmax, ds=(3,3),ignore_border=True,
                                                        st=(1,1),padding=(1, 1),mode='max',
                                                          ) # B x 1 x H x W
                # print('***********'+str(temp2))
                temp3=T.reshape(T.switch(T.eq(temp2, v), v, 0.0),(shape[0],shape[1],-1))

                v_spatial,v_spatial_argmax=T.max_and_argmax(temp3,axis=2,keepdims=True)

                thresh1=T.gt(v_spatial,self.threshold).astype(theano.config.floatX)
# B x C x 1
                thresh2=T.gt(T.reshape(temp2,(shape[0],1,-1)),self.threshold).astype(theano.config.floatX)
# B x 1 x HW
                output_spike=T.reshape((T.eq(T.arange(shape[2]*shape[3]).dimshuffle('x','x',0),v_spatial_argmax)*thresh1),(shape[0],shape[1],shape[2],shape[3]))

                flag=T.ge(thresh1+thresh2,1.0) # B x C x HW
                temp4 = T.reshape(T.switch(flag, self.refractory_voltage,
                                            T.reshape(v, (shape[0],
shape[1], -1))),
                                    (shape[0], shape[1], shape[2], shape[3]))
	        temp3=T.eq(T.arange(self.num_filters).dimshuffle('x',0,'x','x'),arg_max)*v

                self.temp2=temp2
                self.tempy=thresh1
                self.tempx=temp4



                self.output_spike=output_spike
                self.v_out=temp4

	    if self.output_flag==1:

            	return self.output_spike
	    
	    else :
		return self.v_out





        def do_stdp(self):
            self.H_out=self.H_in+self.input
            w_update=stdpOp()(self.output_spike,self.H_out,self.W)
            w_update=T.mean(w_update,axis=0)
            self.update=w_update

            return self.update,self.H_out

        def get_output_shape(self):
            input_shape=lasagne.layers.get_output_shape(self.incoming)
            return (input_shape[0],self.num_filters,input_shape[2],input_shape[3]),input_shape



####################################################################################

class snn_poolLayer(PoolLayer,snn_layer):


        def __init__(self,incoming,stride=6,filter_size=5,refractory_voltage=-np.float32(10000000),threshold=0.99,**kwargs):

           
            self.incoming=incoming
	    self.stdp_enabled=False
            self.threshold=threshold
            self.refractory_voltage=refractory_voltage
	   
            #self.output_spike=T.zeros([1,num_units])
            super(snn_poolLayer,self).__init__(incoming,filter_size,stride)
	

        def get_output_for(self, input, deterministic=False, **kwargs):

            # print(super(snn_denseLayer, self).get_output_for(input, **kwargs))
            self.input=input
            v=self.v_in+super(snn_poolLayer, self).get_output_for(input,**kwargs)
            self.v_in=v
            # vmax=theano.tensor.signal.pool.pool_3d(v, ds=(3,3,self.num_filters), ignore_border=True,
            #                                     st=(1,1,1), padding=(1, 1, 0), mode='max',
            #                                       )

          
            #channelwise
            
	    self.output_spike = T.ge(v,self.threshold)*np.float32(1.0)
	    self.v_out = T.switch(self.output_spike,self.refractory_voltage*np.float32(1.0),v*np.float32(1.0))

	    return self.output_spike





        def do_stdp(self):
            self.H_out=self.H_in+self.input
            w_update=stdpOp()(self.output_spike,self.H_out,self.W)
            w_update=T.mean(w_update,axis=0)
            self.update=w_update

            return self.update,self.H_out

        def get_output_shape(self):
            input_shape=lasagne.layers.get_output_shape(self.incoming)
	    output_shape=lasagne.layers.get_output_shape(self)
            return output_shape,input_shape







###################################################################################




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
        self.time_steps=20
        self.batch_size=self.input_shape[0]


        print 'computing DoG maps ...'


        self.DoG_maps=self.dog_output(input) # the input passed to the class is simply the image, the DoG maps are calculated using
        #this image. A slice of this DoG map  is given as the input to the graph created
        self.get_dog_map=theano.function(inputs=[input],outputs=self.DoG_maps)

        self.input_shape=list(self.input_shape)
        self.input_shape[1]=self.input_shape[1]*2
        self.input_shape=tuple(self.input_shape)

        input_layer = InputLayer(shape = self.input_shape, input_var =  T.reshape(self.DoG_maps[0],self.input_shape))

        self.layers =[
		
		    
		 
                    ('snn_conv1',snn_convLayer,[False,10,4,5,1]), #stdp_enabled,threshold,num_filter,filter_size,output_flag
                    ('pool1', snn_poolLayer, [6,7]),            #stdp_enabled,stride,filter_size
                    ('snn_conv2',snn_convLayer,[False,60,20,15,1]),
                    ('pool1', snn_poolLayer, [2,2]),
                    ('snn_conv3',snn_convLayer,[False,2,10,5,1]),
		    #('gap_layer',GapLayer,[T.max])

                ]
	layer_head = self.layers
        layers=self.layers
        self.layer_names = ['input'] + [attr[0] for attr in layers]

        full_net_layers, _ = self.create_net(layers, input_layer)
        self.full_net_layers=full_net_layers

        self.small_networks=[]
	
	#initial_train,initial_train=create_net(layers) #initial_train not to be used
	
	


        snn_net_train_funcs = []
        snn_net_test_funcs = []



	weight_list=[]
	snn_net_split_on_layer_list = [snn_convLayer, snn_denseLayer]
	
	j=0
	
	if layers[0][1] not in snn_net_split_on_layer_list:
            i = 1
        else:
            i = 0
        for i in range(i, len(layer_head) + 1):
            if i < len(layer_head) and \
                    layers[i][1] in snn_net_split_on_layer_list:
                continue
            tmp_layers = layers[: i]# + layer_tail
	    
	    tmp_layers[-1][2][0] = True
	    #print 'layers ',layers
	    print 'tmp_layers:' , tmp_layers
            print 'Creating: ', [l[0] for l in tmp_layers]
            lyr_list, _ = self.create_net(tmp_layers, input_layer)
            self.small_networks.append(lyr_list)
            self.copy_nets(full_net_layers[-1],
                           lyr_list[-1])
	    '''for k in lyr_list:
		   try :
			print k.W.get_value()
			print 'in small network '+str(j)
		   except:
			print '''

            train,test,_=self.create_snn(lyr_list)
            snn_net_train_funcs.append(train)
            snn_net_test_funcs.append(test)
	    weight_list.append(lyr_list[-1].W)
	    tmp_layers[-1][2][0] = False
	    j=j+1
	    print '*****************************'+'small_network '+str(j)+' created'+'****************************'
	    
	i = len(layer_head) 
	tmp_layers = layers[: i]# + layer_tail
	tmp_layers[-1][2][1] = np.float32(1000000000) #set threshold to infinity
 	tmp_layers[-1][2][-1] = 0 #set output flag to 0 to get voltage


	print 'tmp_layers:' , tmp_layers
	print 'Creating: ', [l[0] for l in tmp_layers]
	lyr_list, _ = self.create_net(tmp_layers, input_layer)
	self.small_networks.append(lyr_list)
	self.copy_nets(full_net_layers[-1],
		   lyr_list[-1])

	_,_,self.get_final_voltage=self.create_snn(lyr_list)
	
	


        self.train_funcs=snn_net_train_funcs
        self.test_funcs=snn_net_test_funcs
	self.weight_list=weight_list
	
	print 'train_funcs' , self.train_funcs
	print 'test_funcs' , self.test_funcs
	

        #self.create_snn()


    def dog_output(self,input_image):

        _,self.channels,self.height,self.width=input_image.shape
        conv_output = T.nnet.conv2d(input_image, dog_W, filter_flip=False,
                                    border_mode='half', subsample=(1, 1))


        conv_output2=conv_output[:,::-1,:,:]

        dog_maps=conv_output - conv_output2
	
        dog_maps=T.ge(dog_maps,0)*dog_maps
	dog_maps = T.switch(T.ge(dog_maps,T.mean(dog_maps)), dog_maps, 0.0)
	#dog_maps=T.set_subtensor(dog_maps[:,1:2,:,:],np.float32(0.0))
	sorted_dog=T.sort(T.reshape(dog_maps,(-1,)))
	    #sorted_dog=T.shape(sorted_dog)-T.sum(T.neq(sorted_dog,0.0))
	num_spikes = T.neq(sorted_dog,0.0)
	num_spikes_per_bin = T.sum( num_spikes )//self.time_steps
	i = T.shape(sorted_dog)[0]- num_spikes_per_bin
	bin_limits=T.zeros(self.time_steps+1)
	bin_limits=T.set_subtensor(bin_limits[0],sorted_dog[-1])
	for j in range(0,self.time_steps):
		bin_limits=T.set_subtensor(bin_limits[j+1],sorted_dog[i])
		i  = i- num_spikes_per_bin

        #return dog_maps,sorted_dog,bin_limits
	#return self.temporal_encoding(dog_maps,bin_limits)
	return T.reshape(self.temporal_encoding(dog_maps,bin_limits)*np.float32(1.0),[self.time_steps,self.batch_size,self.channels*2,self.height,self.width])
	#return output

    def temporal_encoding(self,dog_maps,bin_limits):
	
	def fn(*args):
		print('args')
            	print(args)
		output = T.le(dog_maps,args[0])*np.float32(1.0)*T.gt(dog_maps,args[1])
		return output
	temporal_encoding,_ = theano.scan(fn,sequences=[dict(input= bin_limits, taps = [0,1])], non_sequences=dog_maps,outputs_info=T.zeros_like(dog_maps,dtype=theano.config.floatX))
	#do_encoding = theano.function(inputs=[dog_maps,bin_limits],outputs=temporal_encoding)
	
	#print('compiled')
	#output = do_encoding(dog_maps.eval(),bin_limits.eval())
	return temporal_encoding

    def plot_weights(self,stage_id, file_path = '.', plot_id = 0,
                     max_subplots = 64, max_figures = 64, layer_id = -1,
                     figsize = (6, 6)):
        i = -1
        W=self.weight_list[stage_id]
        #W = T.reshape(W,(1,2,28,28))
        plot_weights(W, 'ID' + str(plot_id) + '_' + 'snn', file_path,
                            max_subplots, max_figures, figsize)
    def copy_nets(self, net1, net2, net1_inputs = [], net2_inputs = []):
        print 'INSIDE: copy_nets'

        def replace_var(obj, var, new_var):
            replaced = False
            for key, value in obj.__dict__.iteritems():
                if value == var:
                    obj.__dict__[key] = new_var
                    replaced = True
            return replaced

        net1_layers = lasagne.layers.get_all_layers(net1, net1_inputs)
        net2_layers = lasagne.layers.get_all_layers(net2, net2_inputs)
        print net2_layers

        lyr_idx = 0
        num_layers_copied = 0
        num_vars_copied = 0
        for lyr_dest in net2_layers:
            if lyr_dest in net2_inputs:
                continue
            if type(net1_layers[lyr_idx]) == type(lyr_dest):
                p_dest_list = lyr_dest.params
                p_src_list = net1_layers[lyr_idx].params
                for p_dest, p_src in zip(p_dest_list, p_src_list):
                    p_dest = p_dest.get_value()
                    if isinstance(p_dest, np.ndarray):
                        print p_dest.shape, p_src.get_value().shape
                        assert(p_dest.shape == p_src.get_value().shape)
                    else:
                        assert(0)

                assert(len(p_dest_list) == len(p_src_list))
                lyr_dest.params = p_src_list
                for p_src, p_dest in zip(p_src_list, p_dest_list):
                    flag = replace_var(lyr_dest, p_dest, p_src)
                    assert(flag)
                    num_vars_copied += 1
                num_layers_copied += 1
                lyr_idx += 1
        print 'Number of variables copied:', num_vars_copied
        print 'Number of layers copied:', num_layers_copied
        print lyr_idx, len(net1_layers), len(net2_layers)
        if lyr_idx != len(net1_layers):
            print 'zWARNING: NOT ALL LAYERS FROM net1 COPIED TO net2.'
        #assert(lyr_idx == len(net1_layers))
        return True

    def create_net(self, layer_details, prev_layer):
        print self.name, '.create_net> building net...'
        layers_list = [prev_layer]
        layers_dict = {'input': prev_layer}
        for attributes in layer_details:
	    print 'attributes : ',attributes
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
    


    def save_model(self, file_name = None, layer_to_save = None):
        if layer_to_save is None:
            layer_to_save = self.layer_list[-1]
        #assert(layer_to_save == self.net['bu_softmax'])
        print 'Saving model starting from layer', layer_to_save.name, '...'
        print 'filename', file_name
        params = lasagne.layers.get_all_param_values(layer_to_save)
        if file_name is not None:
            fp = open(file_name + '.save', 'wb')
            cPickle.dump(params, fp, protocol = cPickle.HIGHEST_PROTOCOL)
            fp.close()
        print 'Done.'
        return params

    def load_model(self, file_name, layer_to_load = None):
        if layer_to_load is None:
            layer_to_load = self.layer_list[-1]
        print 'Loading model starting from layer', layer_to_load.name, '...'
        fp = open(file_name, 'rb')
        params = cPickle.load(fp)
        fp.close()
        lasagne.layers.set_all_param_values(layer_to_load, params)
        print 'Done'




    def create_snn(self,layers):
        print 'Building snn...'
        # if(layers=='None'):
        #     layers=self.layers
        # input_layer = InputLayer(shape = self.input_shape, input_var =  T.reshape(self.DoG_maps[0],self.input_shape)) #the input layer of
        # #the graph which takes a slice of DoG map.
        # all_layers, _ = self.create_net(layers, input_layer)
        all_layers=layers
        # self.all_layers=all_layers
        LR=T.scalar()

        def fn(*args):
            args=list(args)
            print(args)
            print(len(args))
            print('args')
            print(args)
            i=2
            for layer in (all_layers[1:]):
		if(isinstance(layer,snn_layer)):

		        if(layer.stdp_enabled):
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
		if(isinstance(layer,snn_layer)):
		        vH_out_list.append(layer.v_out)
		        if (layer.stdp_enabled):
		            layer.do_stdp()
		            vH_out_list.append(layer.H_out)
		            W_dict.append((layer.W,layer.W+LR*layer.update))
            print('fn returning : ')
            for k in [output_spike_train]+vH_out_list:
		print(k)

            return [output_spike_train]+vH_out_list, W_dict
            #return vH_out_list

        def set_outputs_info():
            output=[]

            # initial_spike_train=T.zeros(all_layers[-1].get_output_shape()[2])
            #initial_spike_train=T.zeros((self.batch_size,32,28,28))
	    initial_spike_train=T.zeros(lasagne.layers.get_output_shape(all_layers[-1]))
            # initial_spike_train=T.zeros((self.batch_size,self.all_layers[-1].num_units))
            #initial_spike_train=T.zeros(self.)
            print(T.shape(initial_spike_train))

            #output.append(initial_spike_train)

            vH_list=[]

            # for layer in all_layers[1:]:
            #     layer.set_inputs(T.vector(),T.tensor4())

            for layer in all_layers[1:]:
		if(isinstance(layer,snn_layer)):
                # print(T.zeros(layer.get_output_shape()[0])
                	vH_list.append(T.zeros(layer.get_output_shape()[0]))
		        if (layer.stdp_enabled) :
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


        delta_weight=T.zeros((1))
        print('*********')
        print(delta_weight)

        for key,value in updates.iteritems():
            delta_weight+=T.mean(abs(value-key))

        delta_weight/=len(updates.keys())

	get_voltage = theano.function(inputs=[self.input,LR],outputs=components[0][-1],updates=updates) #to be used only when output flag is 0


        train=theano.function(inputs=[self.input,LR],outputs=[components[0],delta_weight],updates=updates,on_unused_input='ignore')

        if layers[-1].stdp_enabled==False:
		test=theano.function(inputs=[self.input,LR],outputs=output)
	else :
		print('recursive call')
		layers[-1].stdp_enabled=False
		_,test,_=self.create_snn(layers)


        print('compiled')
	
	return train,test,get_voltage

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





####################################################################################


if __name__ =='__main__' :
   	np.random.seed(64)
	
	batch_size = 1



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
	
	network=snn(data_shape)
	


	#snn_object=snn(data_shape)



