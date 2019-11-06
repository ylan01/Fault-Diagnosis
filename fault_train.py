
from __future__ import print_function

import sys
import os
import time

import numpy as np   
import scipy.io as sio

import theano
import theano.tensor as T

import lasagne

import pickle
import gzip

import binary_net

from collections import OrderedDict

np.random.seed(1234)
#
if __name__ == "__main__":
    
    # BN parameters
    batch_size = 50 # 批训练大小
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # MLP parameters
    num_units = 2048 
    print("num_units = "+str(num_units))
    n_hidden_layers = 3
    print("n_hidden_layers = "+str(n_hidden_layers))
    
    # Training parameters
    num_epochs = 3000   
    print("num_epochs = "+str(num_epochs))
    
    # Dropout parameters
    dropout_in = 0.    # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = 0.2
    print("dropout_hidden = "+str(dropout_hidden))
    
    # BinaryOut
    #activation = binary_net.binary_tanh_unit   
    #print("activation = binary_net.binary_tanh_unit")
    activation = binary_net.binary_sigmoid_unit
    print("activation = binary_net.binary_sigmoid_unit")
    
    # BinaryConnect
    binary = True
    #binary = False
    print("binary = "+str(binary))
    stochastic = False  # 二值化方式：确定性 非随机性
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Decaying LR 
    #LR_start = 0.003  # 学习率 初始值
    LR_start = 0.003
    print("LR_start = "+str(LR_start))
    #LR_fin = 0.000003  # 学习率 最终值
    LR_fin = 0.000003
    print("LR_final = "+str(LR_fin))
    # LR_decay 学习率衰减因子
    LR_decay = (LR_fin/LR_start) ** (1./num_epochs)  # ** 乘方，指数衰减
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    save_path = "fault_b_parameters.npz"
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1  
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading fault dataset...')
    allData = sio.loadmat("fault_Data_0.mat")
  
    train_set_X = allData['trainx'].astype('float32')
    valid_set_X = allData['validx'].astype('float32')
    test_set_X  = allData['testx'].astype('float32')
    
    train_set_y = allData['trainyonehot'].astype('float32')
    valid_set_y = allData['validyonehot'].astype('float32')
    test_set_y  = allData['testyonehot'].astype('float32')
    '''
    train_set_X = 2 * train_set_X - 1
    valid_set_X = 2 * valid_set_X - 1
    test_set_X = 2 * test_set_X - 1
    '''
    train_set_y = 2 * train_set_y - 1
    valid_set_y = 2 * valid_set_y - 1
    test_set_y = 2 * test_set_y - 1
    
    
    print('Building the MLP...') 
 
    # Prepare Theano variables for inputs and targets
    input = T.matrix('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)
    # shape里对应的四个参数分别表示：(batchsize, channels, rows, columns),
    # input_var表示需要连接到网络输入层的theano变量，默认为none
    mlp = lasagne.layers.InputLayer(
            shape=(None,2048),
            input_var=input)

    # 对输入数据加以20%的dropout
    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=dropout_in)

    # k 个隐藏层
    for k in range(n_hidden_layers):
        # 第一层隐层
        mlp = binary_net.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                # Linear activation function \(\varphi(x) = x\)
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)                  
        
        mlp = lasagne.layers.BatchNormLayer(
                mlp,
                epsilon=epsilon, 
                alpha=alpha)

        mlp = lasagne.layers.NonlinearityLayer(
                mlp,
                nonlinearity=activation)
                
        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=dropout_hidden)
    # 输出层
    mlp = binary_net.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)    
    
    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss 损失函数
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_net.compute_grads(loss,mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,mlp)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')

    binary_net.train(
            train_fn,val_fn,
            mlp,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set_X,train_set_y,
            valid_set_X,valid_set_y,
            test_set_X,test_set_y,
            save_path,
            shuffle_parts)

