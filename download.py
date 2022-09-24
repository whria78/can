### HAN SEUNG SEOG (TWITTER = WHRIA78)

import os
import sys
import requests
import concurrent.futures
import numpy as np
import cv2
import time
import wget

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


dest_root=os.path.join(os.getcwd(),'dataset')
try:os.makrdirs(dest_root)
except:pass

def download_wget(image_url):
    img_org=None
    try:
        img_path = wget.download(image_url)
        img_org=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        os.remove(img_path)
    except:
        pass

    return img_org
    
def process_img(img_info_):
    no_,label_,x,y,w,h,image_url=img_info_
    time.sleep(0.1)
    print("Processing : %d / %d" % (no_,no_max))    

    img_org=None
    try:
        img_data = requests.get(image_url,stream=True,headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}).raw

        numpyarray = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
        img_org = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        if img_org is None:
            img_org=download_wget(image_url)

    except:        
        img_org=download_wget(image_url)


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

