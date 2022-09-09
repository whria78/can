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
