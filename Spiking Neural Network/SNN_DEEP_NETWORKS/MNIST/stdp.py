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

from theano import Apply

from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda import CudaNdarray
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)



class stdpOp(GpuOp):



    def __init__(self):

        self.max_threads_dim0 = None


    def make_node(self, output_spike, H_out, weights):
        if output_spike.type.ndim != 4:
            raise TypeError('output_spike must be 4D tensor')
        if H_out.type.ndim != 4:
            raise TypeError('H_out must be 4D tensor')
        if weights.type.ndim != 4:
            raise TypeError('weights must be 4D tensor')
        # if LR.type.ndim != 1:
        #     raise TypeError('LR must be 1D tensor')
        # if weight_update.type.ndim != 4:
        #     raise TypeError('weight_update must be 4D tensor')

        output_spike = as_cuda_ndarray_variable(output_spike)
        H_out = as_cuda_ndarray_variable(H_out)
        weights = as_cuda_ndarray_variable(weights)
        # LR= as_cuda_ndarray_variable(LR)
        #weight_update = as_cuda_ndarray_variable(weight_update)

        print 'MAKENODE: ', output_spike.shape, H_out.shape, weights.shape
        # broadcastable = [output_spike.type.broadcastable[0], H_out.type.broadcastable[0],weights.type.broadcastable[0],
        #                  weight_update,False, False, False, False]
        # otype = CudaNdarrayType(broadcastable=[False] * 4)
        broadcastable=[False,False,False,False,False]
        return Apply(self, [output_spike, H_out, weights], [CudaNdarrayType(broadcastable)()])

    def prepare_node(self, node, storage_map, compute_map, impl):
        super(stdpOp, self).prepare_node(node, storage_map, compute_map, impl)
        print 'IN PREPARE NODE\n'
        if node.op.max_threads_dim0 is None:
            cuda = theano.sandbox.cuda
            device_id = cuda.use.device_number
            if device_id is None:
                cuda.use("gpu",
                         force=False,
                         default_to_move_computation_to_gpu=False,
                         move_shared_float32_to_gpu=False,
                         enable_cuda=False,
                         test_driver=True)
                device_id = cuda.use.device_number
            cuda_ndarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray
            prop = cuda_ndarray.device_properties(device_id)
            node.op.max_threads_dim0 = prop['maxThreadsDim0']


    def c_headers(self):
        return ['cuda_ndarray.cuh', '<stdio.h>']
    '''
    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (0, 23)
    '''
    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        files = ['stdp_kernel.cu']
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read()
                 for f in files]
        return reduce(str.__add__, codes)

    def c_code(self, node, nodename, inp, out_, sub):
        output_spike, H_out, weights = inp
        out, = out_
	max_threads_dim0 = self.max_threads_dim0
	print out


        sub = sub.copy()

        sub.update(locals())
	# print('hello')
        #exit(0)
        return """

    const int *os_size = CudaNdarray_HOST_DIMS(%(output_spike)s);
    const int *h_size = CudaNdarray_HOST_DIMS(%(H_out)s);
    const int *w_size = CudaNdarray_HOST_DIMS(%(weights)s);
    int delta_w_size[5] = {os_size[0], w_size[0], w_size[1], w_size[2], w_size[3]};
    
    if (os_size[1] > %(max_threads_dim0)s)
    {
        fprintf(stderr, "\\nSTDP_OP ERROR: CHANNEL SIZE EXCEEDED THREAD LIMIT (%%d).\\n", %(max_threads_dim0)s);
    }

    Py_XDECREF(%(out)s);

    %(out)s = (CudaNdarray*)CudaNdarray_ZEROS(5,delta_w_size);  //zeros uses int* while ndims uses const int * as second argument
    if (NULL == %(out)s)
    {
        PyErr_Format(PyExc_RuntimeError,
                    "stdpOpMM: Failed to allocate output of %%d x %%d x %%d x %%d",
                    w_size[0], w_size[1], w_size[2], w_size[3]);
    }

    if (!(CudaNdarray_is_c_contiguous(%(output_spike)s) && CudaNdarray_is_c_contiguous(%(H_out)s) \
            && CudaNdarray_is_c_contiguous(%(weights)s) && CudaNdarray_is_c_contiguous(%(out)s)))
    {
        fprintf(stderr, "\\nSTDP_OP ERROR: VARIABLES NOT C-CONTIGUOUS.\\n");
    }

    //dim3 threads(threadx,thready);
    int threads = os_size[1];
    dim3 grid(os_size[0], os_size[2], os_size[3]);

    stdp_kernel <<< grid, threads >>> (%(weights)s->devdata, w_size[0], w_size[1], w_size[2], w_size[3], 
                                        %(output_spike)s->devdata, os_size[0], os_size[1], os_size[2], os_size[3], 
                                        %(H_out)s->devdata, %(out)s->devdata);
    CNDA_THREAD_SYNC;
    cudaError_t sts = cudaGetLastError();
    if (cudaSuccess != sts)
    {
        fprintf(stderr, "\\nSTDP_OP KERNEL ERROR: error_code=%%d, %%s.\\n", sts, cudaGetErrorString(sts));
    }

    //Py_XDECREF(%(out)s);
    if (%(out)s == NULL)
    {
        %(fail)s
    }
""" % sub


if __name__ =='__main__' :
    import numpy as np

    # dict=sio.loadmat('conv_op_test.mat')
    #
    # w=dict('w')
    # os=dict('os')
    # h_out=dict('h_out')
    a = theano.tensor.tensor4()
    b = theano.tensor.tensor4()
    c = theano.tensor.tensor4()
    f = theano.function([a,b,c], stdpOp()(a,b,c))
    print 'compiled'
    #exit(0)
    
    x=1.0*np.ones((32,30,300,400), dtype=np.float32)
    y=1.0*np.ones((32,2,300,400), dtype=np.float32)
    z=0.5*np.ones((30,2,10,10), dtype=np.float32)
    print 'before', datetime.datetime.now()
    out = f(x,y,z)
    print 'after', datetime.datetime.now()
    print 'out shape', out.shape
    print 'out', out
    print 'computed happily!!!'

