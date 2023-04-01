#https://github.com/whria78/can
#Han Seung Seog (whria78@gmail.com)
#https://modelderm.com

import os
import sys
import time
import argparse

#parse arguments
parser = argparse.ArgumentParser(description='CAN5600 & GAN5000 dataset; Han Seung Seog')

parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train (6 by default)')
parser.add_argument('--batch', type=int, default=64, help='batch size (32 by default)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (0.001 by default)')

parser.add_argument('--result_file', type=str, default='result.csv', help='path of the result file')
parser.add_argument('--repeat', type=int, default=10, help='repeat experiment by (20 by default) times')

args = parser.parse_args()

#### TEST configuration ####

# DATA/asantest - ASAN test dataset
# DATA/edin - Edinburgh - !!! Commercial dataset !!!
# DATA/mednoe - MED-NODE dataset
# DATA/pad - PAD-UFES-20 dataset
# DATA/seven - 7-point evaluation criteria dataset
# DATA/snu - subset of SNU dataset
# DATA/water - university of Waterloo dataset

# DATA/gan5000 - GAN5000 dataset (synthetic images = fake)
# DATA/can5600 - CAN5600 dataset (internet images annotated by ModelDerm Build2021)
# DATA/gan5600 - 5600 synthetic images (=GAN5000 dataset)
# DATA/tgan2000 - CAN2000 dataset (manually revised dataset from CAN5600)
# DATA/gan2000 - 2000 synthetic images (=GAN5000 dataset)

all_public="DATA/asantest;DATA/edin;DATA/mednode;DATA/pad;DATA/seven;DATA/snu;DATA/water;"
public_list=["DATA/asantest","DATA/edin","DATA/mednode","DATA/pad","DATA/seven","DATA/snu","DATA/water"]

train_list=["DATA/asantest;DATA/edin;DATA/mednode;DATA/pad;DATA/seven;DATA/snu;DATA/water"]
train_list+=[all_public+"DATA/gan5000",all_public+"DATA/can5600",all_public+"DATA/gan5600",all_public+"DATA/tgan2000",all_public+"DATA/gan2000"]
train_list+=["DATA/gan5000","DATA/can5600","DATA/gan5600","DATA/tgan2000","DATA/gan2000"]


#### RUN
for i in range(0,args.repeat):
    for select_train in train_list:
        for select_test in public_list:
            print(select_train,select_test)
            cmd_=f"python3 train.py --model efficientnet --batch {args.batch} --epoch {args.epoch} --lr {args.lr} --train '{select_train}' --test {select_test} --result {args.result_file}"
            print(cmd_)
            os.system(cmd_)
            time.sleep(1)


