## Clinical photographs Annotated by Neural networks (CAN4000) ##

As for the dermatology public datasets, the Edinburgh dataset, Asan test dataset, SNU subset, ISIC editorial, Fitzpatric 17k, and DDI datasets are available and well-curated. Convolutional neural network (CNN) architecture is commonly used for vision research, but most CNN uses only low-resolution images between 224x224 and 500x500 pixels due to the limited size of GPU memory. For this reason, if the lesional area in a wide-field photograph were small, the characteristic features of the disease could not be identified in the resized photograph. Edinburgh, ASAN test, and SNU subset are the datasets made up of only lesions, but in other datasets, the lesion is needed to be specified in this way to improve the performance of CNNs. However, this process requires a huge amount of time and effort by dermatologists called ‘Data Slave’.

Numerous skin images can be also found on the internet atlas sites and through search engines. But it is impossible for dermatologists to annotate the images at lesinal or intralesional level. In this project, we created a dataset of 4,000 images for melanoma and melanocytic nevus by crawling photographs on the Internet and annotating them by an algorithm (Model Dermatology). Like the way of ImageNet, the dataset consists of labeled internet images. After that, vanilla CNNs had been trained using the created training dataset, and their performance was externally validated using public datasets such as Edinburgh and SNU datasets. 

### How to use ###

Please check dependencies.
<pre><code>pip install wget opencv-python numpy
</code></pre>

The script will download raw images and generate the dataset. Some images may not available for the deletion of the link.
<pre><code>python3 download.py all.csv

or

python3 download.py
</code></pre>

### Malignant melanoma ###

![img](https://github.com/whria78/can/blob/main/thumbnails/malignantmelanoma.jpg?raw=true)

### Other benign melanocytic lesions ###

![img](https://github.com/whria78/can/blob/main/thumbnails/melanocyticnevus.jpg?raw=true)


## Comment ##

Clinical photographs in Dermatology are the most readily available image, however the most non-standardized image. Dermatology images are basically 2D images. However, because the shape of lesion of interest changes according to distance, composition, and clinical course, the actual photograph we encounter is multi-dimensional data. In addition, because the quality of the submitted image is not standardized, any disturbance in the focus and light source could confuse the reader. Despite the need for a much larger amount of data compared to general vision studies, it is difficult for researchers in different hospitals to train an algorithm concurrently while sharing photographs for privacy issues.

The photographs of biopsied cases in appropriate lighting and background is ideal for diagnosis, but most clinical photographs we encounter are not ideal. The problem is that algorithms that are not trained on diverse pictures show poor performance on untrained out-of-distributions (OOD) presented by users. The OOD is not merely limited to untrained diseases. If it is trained solely with pictures that are in focus, the performance of the algorithm may be poor for pictures that are out of focus. Although neural networks in vision research have human-level performance in classifying the images with unified composition, the performance dropped significantly on the photographs with diverse compositions as shown in the ObjectNet study. 

Unlike the objects of ImageNet and ObjectNet, the ground truth of skin disorders is hard to be determined and there is an inter-observers variation that is greater than we usually expect. For the problem of (a) Unclear ground truth, (b) limitation of the lesional resolution, and (c) diverse OODs, it may be impossible to train an algorithm with non-standardized wide-field photographs. In my perspective, the quality of the image should be at least that of mammogram for accurate diagnosis. 

CAN4000 is a dataset of about 4000 training images that consists of melanoma and melanocytic nevus. We collected clinical photographs on the Internet and, we used the detection method (RCNN). Although CNN was robust to massive label noise, the performance of CNN dropped markedly if the size of the training image of each class was less than 1000 [https://arxiv.org/pdf/1705.10694.pdf; Figure 10]. It implies that a large number of data is needed for CNN training, even if it is inaccurate. In addition, in this dataset, we tried to include lesion areas that well reflect the characteristics of the disease. 

In an onychomycosis study, we detected nail plates in wide-field images, cropped the lesional areas, and annotated them to create a large dataset. Even using the training dataset annotated solely by the algorithm, the algorithm trained with the synthetic onychomycosis dataset showed an expert-level performance in the reader test. The proposed method has the advantage of obtaining a dataset with the same disease prevalence whereas the image dataset in hospitals usually consists of specific diseases according to the dermatologist’s interest.

Model Dermatology (ModelDerm; https://modelderm.com) is a classifier that can detect and classify general skin diseases. For diagnosing using only clinical photographs, the performance of the algorithm was comparable with that of specialists. By using region-based CNN, it is possible to detect nodular lesions and process a large number of photographs without the hard work of an annotator. Using the image crawler (https://github.com/whria78/skinimagecrawler), raw images were collected. Using the detection and classification module of the ModelDerm, 5,809 potential crops of melanoma and nevus were extracted. After one Dermatologist examined and excluded inappropriate photographs, a total of 4,456 image crops were finally selected. 

**The limitation of this dataset is that it is annotated by the machine based on image findings. The proposed dataset can be used as an additional training dataset along with the private dataset of hospitals. This dataset is not for validation or testing because of the inaccurate ground truth. Algorithms should be validated using the test dataset with clear ground truth in the intended use setting. Second, different crops from the same image were included, which means that there is a train-test contamination issue. Internal validation shows meaningless exaggerated results.** 


Most curated dataset were compiled with human-centric annotation. The ideal dataset is one that is annotated at a level that teaches babies according to their intended use. Because CNN does not have a common sense, machine-centric annotation may be required. Because retrospective promising results on well curated datasets are often difficult to be reproduced in the real-world setting, data scientists should make efforts to collect and revise the data while repeating training and deployment (https://github.com/whria78/data-in-paper-out).


## Zipped Archive ##

The Dropbox link (275MB) is temporalily available.

https://www.dropbox.com/s/n52y0uqh0063uqt/CAN_4456.tar?dl=0


## Deployment of the Custom Algorithm ##

We released a simple training and web deployment code for testing in the real-world setting.
https://github.com/whria78/data-in-paper-out


## Other Dermatology Dataset (published by us) ##

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


## Other Dermatology Dataset ##

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


## Contributors ##
Seung Seog Han

Soo Ick Cho

Cristian Navarrete-dechent

