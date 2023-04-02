### HAN SEUNG SEOG (TWITTER = WHRIA78)

from geolite2 import geolite2 #pip3 install maxminddb-geolite2


import os
import sys
import requests
import concurrent.futures
import numpy as np
import cv2
import time
import wget

from urllib.parse import urlparse
import socket


csv_file='all.csv'
if len(sys.argv)>1:csv_file=sys.argv[1]
print(csv_file)
f=open(csv_file,'r',encoding='utf-8')

c_info=[]
h_info=[]
while True:
    line=f.readline()
    if not line :break

    split_=line.strip().split(',')

    h_=split_[1]
    c_=split_[2]
    c_info+=[c_]
    h_info+=[h_]

print("Total %d " % (len(c_info)))


from collections import Counter

my_list=c_info
count = Counter(my_list)
for key, value in count.items():
    if value > 1:
        print(f"{key} , {value}")
        

my_list=list(set(h_info))

print(len(my_list)," different host sites")
