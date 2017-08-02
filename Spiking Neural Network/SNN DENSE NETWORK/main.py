
import numpy as np
import theano
import theano.tensor as T
import timeit
#import pickle
import cPickle
import os
import datetime
import cv2
import lasagne
import matplotlib
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
from lasagne.nonlinearities import softmax


def relu1(x):
    return T.switch(x < 0, 0, x)

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from snn import snn,softmax_classifier,count_classifier

import sys


#################################################################

def log(f, txt, do_print = 1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')



#################################################################
def dump_image_batch(X, file_name, max_size = 10, figsize = (10, 10)):
    from itertools import product as iter_product
    nrows = max_size
    ncols = nrows
    if 1 == 1:
        if 1 == 1:
            if 1 == 1:
                shape = X.shape
                figs, axes = plt.subplots(nrows, ncols, figsize = figsize,
                                          squeeze = False)

                for ax in axes.flatten():
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis('off')
                for i, (r, c) in enumerate(iter_product(range(nrows), range(ncols))):
                    if i >= shape[0]:
                        break
                    img = X[i].transpose((1, 2, 0))
                    axes[r, c].imshow(img, interpolation = 'none')

                plt.savefig(os.path.join('', file_name))
                #plt.cla()
                plt.clf()
                plt.close()

#################################################################


def test_classifier(network, dataset,snn_loaded_object):
    loss = 0.0
    accuracy = 0.0
    for i, (X, Y) in enumerate(dataset):

        _, loss_tmp, acc_tmp = network.test(snn_loaded_object.test_batch(X), Y)
        loss += loss_tmp
        accuracy += acc_tmp
    i += 1
    loss /= i
    accuracy /= i
    txt = 'Accuracy: %.4f%%, loss: %.12f, i: %d' \
            % (accuracy * 100.0, loss, i)
    return loss, accuracy, txt
#################################################################
def test_count_classifier(network, dataset,snn_loaded_object):
    # loss = 0.0
    accuracy = 0.0
    for i, (X, Y) in enumerate(dataset):
         acc_tmp = network.test((snn_loaded_object.test_batch(X)>0), Y)
        # loss += loss_tmp
         accuracy += acc_tmp
    i += 1
    # loss /= i
    accuracy /= i
    txt = 'Accuracy: %.4f%%,  i: %d' \
            % (accuracy * 100.0, i)
    return  accuracy, txt
#################################################################
def train_count_classifer(network, datasets, log_path,snn_loaded_object) :
    snapshot_path = os.path.join(log_path, 'snapshots')
    f = open(os.path.join(log_path, 'train.log'), 'w')
    log(f,'count classifer')


    num_epochs = 1
    # losses = np.zeros((2, num_epochs + 1))
    accuracies= np.zeros((2, num_epochs + 1))
    #initial validation loss and accuracy
    # losses[1, 0], accuracies[1, 0], txt = test_classifier(network,
    #                                                  datasets['valid'])
    #
    #
    # log(f, 'TEST valid epoch: ' + str(-1) + ' ' + txt)


    ii = 0
    test_at=[256,625,1250,2500,5000,6250]
    for epoch in range(num_epochs):
        #train_loss = 0.0
        train_accuracy = 0.0
        for i, (X, Y) in enumerate(datasets['train']):
            ii += 1





	    #np.set_printoptions(threshold=np.nan)
	    #print(network.histogram)
	    #print(snn_loaded_object.test_batch(X))

            accuracy = network.train((snn_loaded_object.test_batch(X)>0), Y,ii%200)
            #k=(snn_loaded_object.test_batch(X)>0)

	    # log(f,'Iter: %d [%d]'%(ii,epoch))
	    #print(network.neuron_spiked)
	    #print(network.histogram)
	    #print(network.output)

            #train_loss += loss


            if ii % 200== 0:
                log(f, 'Iter: %d [%d], acc: %.2f%% '
                    % (ii, epoch,accuracy))

	    if ii in test_at :

		log(f, '\nTesting ...')
		_,txt = test_count_classifier(network, datasets['test'],snn_loaded_object)
		log(f, 'epoch: ' + str(epoch) + ' ' + txt)




        # train_loss /= i
        #train_accuracy /= i
        # losses[0, epoch] = train_loss


        epoch += 1



    log(f, '\nTesting at last epoch...')
    _,txt = test_count_classifier(network, datasets['test'],snn_loaded_object)
    log(f, 'epoch: ' + str(epoch) + ' ' + txt)

    log(f, 'Exiting train...')
    f.close()

#################################################################
def train_count_main():
    np.random.seed(11)
    data_path = '/data3/deepak_interns/vikram/vikram/mnist/'
    batch_size = 4

    #data_path = '../data/cifar-100-python'
    model_save_path = './models'


    print 'Loading Dataset'
    mnist = mnist_data_set(data_path, batch_size)
    print 'done loading'


    datasets = mnist.data_sets
    print type(datasets['train'].X)
    print np.max(datasets['train'].X), np.min(datasets['train'].X)
    print datasets['train'].X.shape, datasets['train'].Y.shape
    print datasets['test'].X.shape, datasets['test'].Y.shape
    data_shape = datasets['train'].X.shape
    data_shape = (batch_size, ) + data_shape[1: ]
    print 'Data shape:', data_shape


    path = os.path.join(model_save_path, 'train1')

    print 'loading snn'
    f = open(os.path.join(path, 'snn_autonet' + '.save'), 'rb')
    snn_loaded_object=cPickle.load(f)
    f.close()
    print('Done')

    path = os.path.join(model_save_path, 'classifer')
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'snapshots'))

    print ('creating count classifer')
    classifier=count_classifier()
    print('done')

    print('classifier TRAINING ...')
    train_count_classifer(classifier, datasets, path,snn_loaded_object)
    print('completed training classifer!')
    print('saving classifier...')
    f = open(os.path.join(path, classifier.name + '.save'), 'wb')
    #theano.misc.pkl_utils.dump()
    sys.setrecursionlimit(50000)
    cPickle.dump(classifier, f, protocol = cPickle.HIGHEST_PROTOCOL)
    f.close()
    print('Done')
#################################################################
#################################################################
def train_classifer(network, datasets, log_path,snn_loaded_object):
    snapshot_path = os.path.join(log_path, 'snapshots')
    LR = 10
    f = open(os.path.join(log_path, 'train.log'), 'w')
    log(f,'softmax classifer')
    log(f, 'Learning rates LR: %f ' % (LR))

    num_epochs = 20
    losses = np.zeros((2, num_epochs + 1))
    accuracies= np.zeros((2, num_epochs + 1))
    #initial validation loss and accuracy
    # losses[1, 0], accuracies[1, 0], txt = test_classifier(network,
    #                                                  datasets['valid'])
    #
    #
    # log(f, 'TEST valid epoch: ' + str(-1) + ' ' + txt)


    ii = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        for i, (X, Y) in enumerate(datasets['train']):
            ii += 1


            # network.train(X, Y, LR)
            loss, accuracy = network.train(snn_loaded_object.test_batch(X), Y, LR)

            train_loss += loss
            train_accuracy += accuracy
            if ii % 200 == 0:
                log(f, 'Iter: %d [%d], loss: %f, acc: %.2f%%, '
                    'avg_loss: %f, avg_acc: %.2f%%'
                    % (ii, epoch, loss, accuracy, train_loss / (i + 1),
                       100.0 * (train_accuracy / (i + 1))))

        train_loss /= i
        train_accuracy /= i
        losses[0, epoch] = train_loss
        accuracies[0, epoch] = train_accuracy

        log(f, '\nEpoch %d: avg_Loss: %.12f, avg_Acc: %.12f'
            % (epoch, train_loss, train_accuracy * 100.0))

        epoch += 1

        losses[1, epoch], accuracies[1, epoch], txt = test_classifier(network,
                                                    datasets['valid'],snn_loaded_object)
        log(f, 'TEST valid epoch: ' + str(epoch) + ' ' + txt)



        p1, = plt.plot(losses[0, : epoch],label='Training loss')
        p2, = plt.plot(losses[1, : epoch],label='Validation loss')
        plt.legend()
        plt.ylabel('LOSS')
        plt.xlabel('EPOCH NUMBER')
        plt.savefig(os.path.join(snapshot_path, 'losses.jpg'))
        plt.clf()
        plt.close()

        p1, = plt.plot(100*accuracies[0, : epoch],label='Training accuracy')
        p2, = plt.plot(100*accuracies[1, : epoch],label='Validation accuracy')
        plt.legend()
	plt.ylabel('ACCURACY %')
        plt.xlabel('EPOCH NUMBER')
        plt.savefig(os.path.join(snapshot_path, 'accuracy.jpg'))
        plt.clf()
        plt.close()


    np.save(os.path.join(snapshot_path, 'losses.npy'), losses)
    '''
    min_epoch = np.argmin(losses)
    log(f, 'Done Training.\n Minimum loss %f at epoch %d' %
        (losses[min_epoch], min_epoch))
    # '''
    log(f, '\nTesting at last epoch...')
    _, _, txt = test_classifier(network, datasets['test'],snn_loaded_object)
    log(f, 'epoch: ' + str(epoch) + ' ' + txt)
    log(f, 'Exiting train...')
    f.close()
    return
#################################################################
def train_snn(network, datasets, log_path):
    snapshot_path = os.path.join(log_path, 'snapshots')
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    LR = 1.0e-2
   
    f = open(os.path.join(log_path, 'train.log'), 'w')
    log(f, 'Learning rates LR: %f ' % (LR))

    num_epochs = 80
    all_dw=np.zeros(num_epochs+1)
    ii = 0
    for epoch in range(num_epochs):
        delta_weight=0

        for i, (X, Y) in enumerate(datasets['train']):
            # if(i==400):
            #     break
            ii += 1

            output_spike_last_layer,delta_w = network.train(X,LR)
            #print(delta_w)
            delta_weight+=delta_w
            assert(np.amin(network.all_layers[-1].W.eval())>0)
            assert(np.amax(network.all_layers[-1].W.eval())<1)
            if ii % 2000 == 0:
                #x=np.argmax(np.squeeze(output_spike_last_layer))
                #log(f,'Iter: %d [%d],spiked in time step : %d , neuron : %d, delta_weight_mean : %0.8e'%(ii, epoch,np.floor_divide(x,1024),np.mod(x,1024),delta_weight/(i+1.0)))
		 log(f,'Iter: %d [%d], delta_weight_mean : %0.8e'%(ii, epoch,delta_weight/(i+1.0)))

		

        delta_weight/=i

        all_dw[epoch]=delta_weight
        epoch += 1

        log(f,'plotting graph')
        p1, = plt.plot(all_dw[:epoch],label='mean weight change')
        #plt.legend()
        plt.ylabel('MEAN weight change')
        plt.xlabel('EPOCH NUMBER')
        plt.savefig(os.path.join(snapshot_path, 'mean_weight_change.jpg'))
        plt.clf()
        plt.close()

        if (epoch % 2 == 0):
            log(f, 'Plotting weights...')
            network.plot_weights('./models/plots/weights', epoch)
        if(epoch%20==0):
            log(f,'testing validation set')
            log(f,'training count classifer')

            num_epochs_2 = 1
            classifier=count_classifier()
            accuracies= np.zeros((2, num_epochs + 1))
            iii=0
            for epoch_2 in range(num_epochs_2):
                #train_loss = 0.0
                train_accuracy = 0.0
                for i, (X, Y) in enumerate(datasets['train']):
                    iii += 1

                    if(iii==10001):
                        break


                    accuracy = classifier.train((network.test_batch(X)>0), Y,iii%5)

                    if iii % 1000== 0:
                        log(f, 'Iter: %d [%d], acc: %.2f%% '
                            % (iii, epoch_2,accuracy))



		log(f, '\nTesting validation')
		_,txt = test_count_classifier(classifier, datasets['valid'],network)
		log(f, 'epoch: ' + str(epoch_2) + ' ' + txt)




                # train_loss /= i
                #train_accuracy /= i
                # losses[0, epoch] = train_loss


                epoch_2 += 1



            print('saving weights')
            fp = open(snapshot_path+str(epoch) + '.save', 'wb')
            cPickle.dump(network.all_layers[-1].W.eval(), fp, protocol = cPickle.HIGHEST_PROTOCOL)
            fp.close()
            print 'Done.'


    f.close()
    return


#################################################################

#kind of main function
def train_snn_main():

    np.random.seed(11)
    data_path = '/data3/deepak_interns/vikram/vikram/mnist/'
    batch_size = 4

    #data_path = '../data/cifar-100-python'
    model_save_path = './models'


    print 'Loading Dataset'
    mnist = mnist_data_set(data_path, batch_size)
    print 'done loading'


    datasets = mnist.data_sets
    print type(datasets['train'].X)
    print np.max(datasets['train'].X), np.min(datasets['train'].X)
    print datasets['train'].X.shape, datasets['train'].Y.shape
    print datasets['test'].X.shape, datasets['test'].Y.shape
    print datasets['valid'].X.shape, datasets['valid'].Y.shape
    data_shape = datasets['train'].X.shape
    data_shape = (batch_size, ) + data_shape[1: ]
    print 'Data shape:', data_shape

    print 'Creating snn'
    network = snn(list(data_shape))

    assert(np.amin(network.all_layers[-1].W.eval())>0)
    assert(np.amax(network.all_layers[-1].W.eval())<1)
    # print(np.amin(network.all_layers[-1].W.eval()))

    path = os.path.join(model_save_path, 'train1')
    if not os.path.exists(path):
        os.makedirs(path)
    # path = os.path.join(model_save_path, 'snapshots')
    # if not os.path.exists(path):
    #     os.makedirs(path)
    print('SNN TRAINING ...')
    train_snn(network, datasets, path)
    print('completed training snn !')
    print('saving trained snn...')
    f = open(os.path.join(path, network.name + '.save'), 'wb')
    #theano.misc.pkl_utils.dump()
    sys.setrecursionlimit(50000)
    cPickle.dump(network, f, protocol = cPickle.HIGHEST_PROTOCOL)
    f.close()
    print('Done')
#################################################################
def train_classifer_main():

    np.random.seed(11)
    data_path = '/data3/deepak_interns/vikram/vikram/mnist/'
    batch_size = 128

    #data_path = '../data/cifar-100-python'
    model_save_path = './models'


    print 'Loading Dataset'
    mnist = mnist_data_set(data_path, batch_size)
    print 'done loading'


    datasets = mnist.data_sets
    print type(datasets['train'].X)
    print np.max(datasets['train'].X), np.min(datasets['train'].X)
    print datasets['train'].X.shape, datasets['train'].Y.shape
    print datasets['test'].X.shape, datasets['test'].Y.shape
    print datasets['valid'].X.shape, datasets['valid'].Y.shape
    data_shape = datasets['train'].X.shape
    data_shape = (batch_size, ) + data_shape[1: ]
    print 'Data shape:', data_shape


    path = os.path.join(model_save_path, 'train1')

    print 'loading snn'
    f = open(os.path.join(path, 'snn_autonet' + '.save'), 'rb')
    snn_loaded_object=cPickle.load(f)
    f.close()
    print('Done')

    path = os.path.join(model_save_path, 'classifer')
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'snapshots'))

    print ('creating classifer')
    classifier=softmax_classifier()
    # print('loading classifier')
    # f = open(os.path.join(path, 'softmax_classifer' + '.save'), 'rb')
    # classifier=cPickle.load(f)
    #f.close()
    #print('done')


    print('classifier TRAINING ...')
    train_classifer(classifier, datasets, path,snn_loaded_object)
    print('completed training classifer!')
    print('saving classifier...')
    f = open(os.path.join(path, classifier.name + '.save'), 'wb')
    #theano.misc.pkl_utils.dump()
    sys.setrecursionlimit(50000)
    cPickle.dump(classifier, f, protocol = cPickle.HIGHEST_PROTOCOL)
    f.close()
    print('Done')
#################################################################


if __name__ == '__main__':
    #train_snn_main()
    train_count_main()
