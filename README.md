### Clinical photographs Annotated for Neural network training (CAN10000) ###

As for the dermatology public datasets, the Edinburgh dataset, Asan test dataset, SNU subset, ISIC editorial, SD, Fitz 17k, etc. are publicly available. A lot of images are also found on the internet atlas site and search engines. Convolutional neural network (CNN) architecture is commonly used for vision research, but most CNN uses only low-resolution images between 224x224 and 500x500 pixels due to the lack of GPU memory. For this reason, if the lesional area in a wide-field photograph is small, the characteristic image findings of the disease could not be identified in the resized photograph. Edinburgh, ASAN test, and SNU subset are curated datasets made up of only lesions, but in other datasets, the lesions are needed to be specified in this way to improve the performance of CNNs. However, this process requires a huge amount of time and effort by dermatologists called ‘data slaves’.

In this project, we created a dataset of 10,000 images for 10 diseases (=10 diagnoses of Edinburgh dataset) by crawling photographs on the Internet and annotating them by an algorithm. Like the way of ImageNet, the dataset consists of labeled internet images. After that, vanilla CNNs had been trained using the created training dataset, and their performance was externally validated using public datasets such as Edinburgh and SNU datasets. In addition, using GAN technology, another synthetic dataset was generated and we also investigated the performance of the algorithm trained with the GAN images.

### Actinic keratosis ###

![img](https://github.com/whria78/can/blob/main/thumbnails/actinickeratosis.jpg?raw=true)

### Basal cell carcinoma ###

![img](https://github.com/whria78/can/blob/main/thumbnails/basalcellcarcinoma.jpg?raw=true)

### Dermatofibroma ###

![img](https://github.com/whria78/can/blob/main/thumbnails/dermatofibroma.jpg?raw=true)

### Hemangioma ###

![img](https://github.com/whria78/can/blob/main/thumbnails/hemangioma.jpg?raw=true)

### Malignant melanoma ###

![img](https://github.com/whria78/can/blob/main/thumbnails/malignantmelanoma.jpg?raw=true)

### Melanocytic nevus ###

![img](https://github.com/whria78/can/blob/main/thumbnails/melanocyticnevus.jpg?raw=true)

### Pyogenic granuloma ###

![img](https://github.com/whria78/can/blob/main/thumbnails/pyogenicgranuloma.jpg?raw=true)

### Seborrheic keratosis ###

![img](https://github.com/whria78/can/blob/main/thumbnails/seborrheickeratosis.jpg?raw=true)

### Squamous cell carcinoma ###

![img](https://github.com/whria78/can/blob/main/thumbnails/squamouscellcarcinoma.jpg?raw=true)

### Intraepithelial carcinoma ###


### Discussion ###

Clinical photographs in Dermatology are the most readily available image, however the most non-standardized image. Dermatology images are basically 2D images. However, because the shape of lesion of interest changes according to distance, composition, and clinical course, the actual photograph we encounter is multi-dimensional data. In addition, because the quality of the submitted image is not standardized, any disturbance in the focus and light source could confuse the reader. Since there are more than 100 conditions to be considered in dermatological diseases, it is also challenging to make an algorithm that can handle multiple classes. Despite the need for a much larger amount of data compared to general vision studies, it is difficult for researchers in different hospitals to train an algorithm concurrently while sharing patients’ photographs for privacy issues.

A biopsied case with photographs in appropriate lighting and background is ideal for diagnosis, but most clinical photographs we encounter are not ideal. The problem is that algorithms that are not trained on diverse pictures show poor performance on untrained out-of-distributions (OOD) presented by users. The OOD is not merely limited to untrained diseases. If it is trained solely with pictures that are in focus, the performance of the algorithm may be poor for pictures that are out of focus. Although neural networks in vision research have human-level performance in classifying the images with unified composition, the performance dropped significantly on the photographs with diverse compositions as shown in the ObjectNet study.

In an onychomycosis study, we detected nail plates in wide-field images, cropped the lesional areas, and annotated them to create a large dataset. Even using the training dataset annotated solely by the algorithm, the algorithm trained with the virtual dataset showed an expert-level performance in the reader test. The proposed method has the advantage of obtaining a dataset with the same disease prevalence whereas the image dataset in hospitals usually consists of specific diseases according to the dermatologist’s interest.

Model dermatology is a multi-class classifier that can classify 184 diseases. For diagnosing using only clinical photographs, the performance of the algorithm was comparable with that of specialists. By using region-based CNN, it is possible to detect nodular lesions and process a large number of photographs without the hard work of an annotator.

CAN10000 is a dataset of 10000 training images for detecting nodular skin lesions. We collected clinical photographs on the Internet. The limitation of this dataset is that it is annotated by the machine based on image findings. We expect a better algorithm if the proposed dataset is used as an adjuvant dataset along with the private dataset of hospitals. Algorithms should be validated in the intended use setting, using the test dataset with clear ground truth.
