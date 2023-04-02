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

import threading
file_lock=threading.Lock()


img_info=[]
csv_file='all.csv'
if len(sys.argv)>1:csv_file=sys.argv[1]
f=open(csv_file,'r',encoding='utf-8')
no_max=0
while True:
    line=f.readline()
    if not line :break

    if len(line)>2:
        if line[0:2]=='##':continue

    split_=line.strip().split(' ')

    no_=int(split_[0])
    if no_max<no_:no_max=no_
    label_=split_[1]
    x=float(split_[2])
    y=float(split_[3])
    w=float(split_[4])
    h=float(split_[5])
    url=split_[6]

    img_info+=[(no_,label_,x,y,w,h,url)]

print("Total %d urls" % (len(img_info)))

result_string=[]
def process_img(img_info_):
    no_,label_,x,y,w,h,image_url=img_info_
    print("Processing : %d / %d" % (no_,no_max))    

    global result_string

    try:
        parsed_url = urlparse(image_url)
        host = parsed_url.netloc
        ip_address = socket.gethostbyname(host)
        country_=geolite2.reader().get(ip_address)['country']['names']['en']

        result_string+=[image_url+","+host+","+country_+"\n"]
    except:
        host="-"
        country_='-'
        result_string+=[image_url+","+host+","+country_+"\n"]



#for img_info_ in img_info:
#    process_img(img_info_)

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    future_to={executor.submit(process_img,img_info_):img_info_ for img_info_ in img_info}

print(len(result_string))
f=open("result.csv","a",encoding='utf-8')    
for result_string_ in result_string:
    f.write(result_string_)