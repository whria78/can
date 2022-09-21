### HAN SEUNG SEOG (TWITTER = WHRIA78)

import os
import sys
import requests
import concurrent.futures
import numpy as np
import cv2
import time

img_info=[]
csv_file=''
if len(sys.argv)>1:csv_file=sys.argv[1]
f=open(csv_file,'r',encoding='latin-1')
no_=0
while True:
    line=f.readline()
    if not line :break

    if len(line)>2:
        if line[0:2]=='##':continue

    split_=line.strip().split(' ')

    label_=split_[0]
    x=float(split_[1])
    y=float(split_[2])
    w=float(split_[3])
    h=float(split_[4])
    url=split_[5]

    img_info+=[(no_,label_,x,y,w,h,url)]
    no_+=1    

print("Total %d urls" % (len(img_info)))


dest_root=os.path.join(os.getcwd(),'dataset')
try:os.makrdirs(dest_root)
except:pass
  
    
def process_img(img_info_):
    no_,label_,x,y,w,h,image_url=img_info_
    time.sleep(0.1)
    print("Processing : %d / %d" % (no_,len(img_info)))    

    try:
        img_data = requests.get(image_url,stream=True,headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}).raw
    except:
        print("Failed : ",image_url)
        return;

    numpyarray = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
    img_org = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    if img_org is None:
        print("Failed : ",image_url)
        return;

    height,width,channel=img_org.shape

    x1=int((x-w/2)*width)
    x2=int((x+w/2)*width)
    y1=int((y-h/2)*height)
    y2=int((y+h/2)*height)

    dest_path=os.path.join(dest_root,label_,'%s_%s.png' % (str(no_).zfill(5),label_))
    try:os.makedirs(os.path.dirname(dest_path))
    except:pass


    cv2.imwrite(dest_path,img_org[y1:y2,x1:x2])


#for img_info_ in img_info:
#    process_img(img_info_)
#sys.exit(1)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    future_to={executor.submit(process_img,img_info_):img_info_ for img_info_ in img_info}

