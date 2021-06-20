import numpy as np
import os
import sys
train_list_path="./T-Brain/TrainDataset/train_list.txt"

f = open('train_list_path', 'a')
for i in range(1,4001):
    f.write("img_"+str(i)+".jpg")
    f.write('\n')
f.close()    
