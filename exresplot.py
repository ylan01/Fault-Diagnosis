# -*- coding: utf-8 -*-
"""
Created on Thu Aug 09 16:00:01 2018

@author: HU
"""
import numpy as np
import matplotlib.pyplot as plt

trainloss = []
valloss = []

with open('trainloss.txt', 'r') as tr:
    for line in tr:
        trainloss.append(list(map(float,line.split(','))))


with open('valloss.txt', 'r') as tr:
    for line in tr:
        valloss.append(list(map(float,line.split(','))))

epoch = range(1, 107)
trainloss = np.array(trainloss)
     
plt.figure()
l1, = plt.plot(epoch, trainloss, 
         color='red',   # 线颜色
         linewidth=1.0,  # 线宽 
         linestyle='--'  # 线样式
        )
l2, = plt.plot(epoch, valloss, 
         color='blue',   # 线颜色
         linewidth=1.0,  # 线宽 
         linestyle='-'  # 线样式
        )

plt.xlim((1, 107))
plt.xlabel("epoch")
plt.ylabel("Loss")

plt.text(50, 0.2, 
         r'$fluctuating\ downward$',
         fontdict={'size':15,'color':'r'})



plt.legend(handles=[l1, l2], 
           labels = ['train_loss', 'val_loss'], 
           loc = 'best'
          )


plt.show()
