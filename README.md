## Clinical photographs Annotated by Neural networks (CAN5600 and CAN2000 dataset) ##

![img](https://github.com/whria78/can/blob/main/thumbnails/melanoma_nevus.jpg?raw=true)

The lesion is needed to be specified in this way to improve the performance of CNNs. However, this process requires a huge amount of time and effort by dermatologists called ‘Data Slave’. In this project, we created a dataset of 5,619 images for melanoma and melanocytic nevus by crawling photographs on the Internet and annotating them by an algorithm (Model Dermatology; https://modelderm.com). Like the way of ImageNet, this dataset consists of labeled internet images. 

## Data repository

| Path | Description
| :--- | :----------
| [Main](https://github.com/whria78/can/) | Main directory
| &ensp;&ensp;&boxvr;&nbsp; [DATASET](https://github.com/whria78/can/tree/main/DATASET) | Dataset
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; CAN2000.csv | CAN2000 dataset (URL, crop location)
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; CAN5600.csv | CAN5600 dataset (URL, crop location)
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; LESION130k.csv | LESION130k dataset (URL, crop location)
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; download.py | Download script
| &ensp;&ensp;&boxvr;&nbsp; [SCRIPTS](https://github.com/whria78/can/tree/main/SCRIPTS) | Script
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [DATA](https://github.com/whria78/can/tree/main/SCRIPTS/DATA) | Training and Test datasets
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [asantest](https://github.com/whria78/can/tree/main/SCRIPTS/DATA/asantest) | Asan test dataset
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [snu](https://github.com/whria78/can/tree/main/SCRIPTS/DATA/snu) | subset of SNU dataset
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [gan2000](https://github.com/whria78/can/tree/main/SCRIPTS/DATA/gan2000) | GAN2000
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [gan5000](https://github.com/whria78/can/tree/main/SCRIPTS/DATA/gan5000) | GAN5000
| &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [gan5600](https://github.com/whria78/can/tree/main/SCRIPTS/DATA/gan5600) | GAN5600
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [out_source_0513](https://github.com/whria78/can/tree/main/SCRIPTS/out_source_0513) | An example of projecting images to latent space (nevus; seed0513).
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [out_source_1119](https://github.com/whria78/can/tree/main/SCRIPTS/out_source_1119) | An example of projecting images to latent space (melanoma; seed1119).
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; morph.py | Morphing script
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; project.py | project.py of STYLEGAN2-ADA-PYTORCH
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; run_all.py | Running all configurations (Training EfficientNet and Testing)
| &ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; train.py | Training EfficientNet & Testing with various datasets

## 1. Download CAN5600 / CAN2000 / LESION130k datasets ##

This repository contains only the download URLs of datasets. The zipped archived can be requested by email (whria78@gmail.com)

Please check dependencies.
```.bash
pip install wget opencv-python numpy
```

The script will download raw images and generate the dataset. Some images may not available for the deletion of the link.
```.bash
# CAN5600
python3 download.py CAN5600.csv
# CAN2000
python3 download.py CAN2000.csv
# LESION130k
python3 download.py LESION130k.csv
```


## 2. Download Public Datasets ##

PAD-UFES-20 (6 tumorous disorders; 2,298  images)
https://data.mendeley.com/datasets/zr7vgbcyr2/1
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7479321/

MED-NODE (melanoma and nevus; 170 images)
https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/

7-point criteria evaluation database (melanoma and melanocytic disorders; 2,045 images)
https://derm.cs.sfu.ca/Welcome.html

Skin Cancer Detection dataset of university of Waterloo
https://uwaterloo.ca/vision-image-processing-lab/research-demos/skin-cancer-detection

Edinburgh Dermofit (10 nodular disorders; 1,300 images)
https://licensing.edinburgh-innovations.ed.ac.uk/product/dermofit-image-library


cf) The subset of SNU and ASAN test datasets are included in this repository.


## 3. Training GAN Models ##

For the GAN training, StyleGAN2-ADA configuration in the StyleGAN3 was used (https://github.com/NVlabs/stylegan3).

```.bash
# Data Preparation
# Scaled down 512x512 resolution using dataset_tool.py 
python3 dataset_tool.py --source=[DATA/LESION130k] --dest=[DATA/LESION130k_512] --resolution=512x512
python3 dataset_tool.py --source=[DATA/CAN2000/malignantmelanoma] --dest=[DATA/CAN2000_mel512] --resolution=512x512
python3 dataset_tool.py --source=[DATA/CAN2000/melanocyticnevus] --dest=[DATA/CAN2000_n512] --resolution=512x512
python3 dataset_tool.py --source=[DATA/CAN2000] --dest=[DATA/CAN2000_512] --resolution=512x512


# Pretrain
# Training for the Pretrain GAN Model of General Skin Disorders
python3 train.py --outdir=training-runs --data=[DATA/LESION130k_512] --mirror=1 --gpus=2 --gamma=8.2 --cfg=stylegan2 --batch=16 --batch-gpu=8 --map-depth=2 --glr=0.003 --dlr=0.003 --resume=ffhq512.pkl --kimg=10000 --snap=10 

python3 gen_images.py --network=c10000.pkl --seeds=0-9999 --outdir=c10000


# Training and Generating GAN5000 dataset
python3 train.py --outdir=training-runs --data=[DATA/CAN2000_mel512] --mirror=1 --gpus=1 --gamma=32 --cfg=stylegan2 --kimg=500 --snap=1 --map-depth=2 --batch=16 --batch-gpu=8 --glr=0.003 --dlr=0.003 --resume=c10000.pkl --freezed=13

python3 gen_images.py --network=m500.pkl --seeds=0-2500 --outdir=[GAN5000/malignantmelanoma]

python3 train.py --outdir=training-runs --data=[DATA/CAN2000_n512] --mirror=1 --gpus=1 --gamma=32 --cfg=stylegan2 --kimg=500 --snap=1 --map-depth=2 --batch=16 --batch-gpu=8 --glr=0.003 --dlr=0.003 --resume=c10000.pkl --freezed=13

python3 gen_images.py --network=n500.pkl --seeds=0-2500 --outdir=[GAN5000/melanocyticnevus]


# Training for Morphing
python3 train.py --outdir=training-runs --data=[DATA/CAN2000_512] --mirror=1 --gpus=2 --gamma=32 --cfg=stylegan2 --kimg=500 --snap=1 --map-depth=2 --batch=32 --batch-gpu=8 --glr=0.003 --dlr=0.003 --resume=c10000.pkl --freezed=13; 

# Projecting images to latent space.
# To get the morphing images, we used the project.py in the StyleGAN2-ADA (https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/projector.py).
python3 projector.py --save-video 0 --num-steps 1000 --outdir=out_source_0513 --target=seed0513.jpg --network=mn500.pkl
python3 projector.py --save-video 0 --num-steps 1000 --outdir=out_source_1119 --target=seed1119.jpg --network=mn500.pkl

# Generating the morphing images.
python3 morph.py --network=mn500.pkl --source=out_source_0513/projected_w.npz --target=out_source_1119/projected_w.npz

```

Trained GAN Models are available at https://doi.org/10.6084/m9.figshare.21507189


## 3. Training CNN Models ##

Please check the test folders - [DATA/asan], [DATA/snu], [DATA/pad], [DATA/seven], [DATA/water], [DATA/edin], [DATA/mednode]

Edinburgh dataset [DATA/edin] is a commercial dataset

All images should be squared off with cropped edges.

```.bash
# An example of training a EfficientLite0 with GAN5000 & Testing with SNU dataset
python3 train.py --model efficientnet --opt radam --batch 64 --epoch 30 --lr 0.001 --train [DATA/GAN5000] --test [DATA/snu] --result log.txt

# An example of running all test configurations
python3 run_all.py --result_file log.txt
```

## Deployment Tool for the Custom Algorithm ##

We released a simple training and web deployment code for testing in the real-world setting.
https://github.com/whria78/data-in-paper-out


## List of Dermatology Datasets (Clinical Photographs) ##

SNU Test dataset (general disorders; 240 images)
https://figshare.com/articles/dataset/SNU_SNU_MELANOMA_and_Reddit_dataset_Quiz/6454973

Asan Test dataset (10 nodular disorders; 1,276 Asan + 152 Hallym images)
https://figshare.com/articles/software/Caffemodel_files_and_Python_Examples/5406223

Model Onychomycosis (onychomycosis, nail dystrophy; 1,358 images)
https://figshare.com/articles/dataset/Model_Onychomycosis_Training_Datasets_JPG_thumbnails_and_Validation_Datasets_JPG_images_/5398573?file=9302506

Model Onychomycosis, Virtual dataset (6 nail disorders; 3,317 images)
https://figshare.com/articles/dataset/Virtual_E_Dataset/5513407

RD Dataset (Reddit melanoma community images; 1,282 images)
https://figshare.com/articles/dataset/RD_Dataset/15170853

PAD-UFES-20 (6 tumorous disorders; 2,298  images)
https://data.mendeley.com/datasets/zr7vgbcyr2/1
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7479321/

Edinburgh Dermofit (10 nodular disorders; 1,300 images)
https://licensing.edinburgh-innovations.ed.ac.uk/product/dermofit-image-library

MED-NODE (melanoma and nevus; 170 images)
https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/

SD-198 (general disorders; 6,584 images)
https://xiaoxiaosun.com/docs/2016-eccv-sd198.pdf
cf) DermQuest, Galderma

Diverse Dermatology Images (general disorders; 656 images)
https://ddi-dataset.github.io/

SKINCON (general disorders; 3,230 images)
https://skincon-dataset.github.io/

Fitzpatrick 17k (general disorders; 16,577 images)
https://github.com/mattgroh/fitzpatrick17k
cf) https://www.dermaamin.com/site/  http://atlasdermatologico.com.br/

7-point criteria evaluation database (melanoma and melanocytic disorders; 2,045 images)
https://derm.cs.sfu.ca/Welcome.html

Skin Cancer Detection dataset of university of Waterloo
https://uwaterloo.ca/vision-image-processing-lab/research-demos/skin-cancer-detection

SKINL2 dataset (light field dataset of skin lesions)
https://www.it.pt/AutomaticPage?id=3459