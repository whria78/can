## Clinical photographs Annotated by Neural networks (CAN5000) ##

As for the dermatology public datasets, the Edinburgh dataset, Asan test dataset, SNU subset, ISIC editorial, Fitzpatric 17k, and DDI datasets are available and well-curated. Convolutional neural network (CNN) architecture is commonly used for vision research, but most CNN uses only low-resolution images between 224x224 and 500x500 pixels due to the limited size of GPU memory. For this reason, if the lesional area in a wide-field photograph were small, the characteristic features of the disease could not be identified in the resized photograph. Edinburgh, ASAN test, and SNU subset are the datasets made up of only lesions, but in other datasets, the lesion is needed to be specified in this way to improve the performance of CNNs. However, this process requires a huge amount of time and effort by dermatologists called ‘Data Slave’.

Numerous skin images can be also found on the internet atlas sites and through search engines. But it is impossible for dermatologists to annotate the images at lesinal or intralesional level. In this project, we created a dataset of 5,000 images for melanoma and melanocytic nevus by crawling photographs on the Internet and annotating them by an algorithm (Model Dermatology). Like the way of ImageNet, the dataset consists of labeled internet images. After that, vanilla CNNs had been trained using the created training dataset, and their performance was externally validated using public datasets such as Edinburgh and SNU datasets. In addition, using GAN technology, another synthetic dataset was generated. We also investigated the performance of the algorithm trained with the sythetic images.

### How to use ###
<pre><code>
python3 download.py
</code></pre>

### Malignant melanoma ###

![img](https://github.com/whria78/can/blob/main/thumbnails/malignantmelanoma.jpg?raw=true)

### Other benign melanocytic lesions ###

![img](https://github.com/whria78/can/blob/main/thumbnails/melanocyticnevus.jpg?raw=true)


## Comment ##

Clinical photographs in Dermatology are the most readily available image, however the most non-standardized image. Dermatology images are basically 2D images. However, because the shape of lesion of interest changes according to distance, composition, and clinical course, the actual photograph we encounter is multi-dimensional data. In addition, because the quality of the submitted image is not standardized, any disturbance in the focus and light source could confuse the reader. Despite the need for a much larger amount of data compared to general vision studies, it is difficult for researchers in different hospitals to train an algorithm concurrently while sharing photographs for privacy issues.

The photographs of biopsied cases in appropriate lighting and background is ideal for diagnosis, but most clinical photographs we encounter are not ideal. The problem is that algorithms that are not trained on diverse pictures show poor performance on untrained out-of-distributions (OOD) presented by users. The OOD is not merely limited to untrained diseases. If it is trained solely with pictures that are in focus, the performance of the algorithm may be poor for pictures that are out of focus. Although neural networks in vision research have human-level performance in classifying the images with unified composition, the performance dropped significantly on the photographs with diverse compositions as shown in the ObjectNet study. 

Unlike the objects of ImageNet and ObjectNet, the ground truth of skin disorders is hard to be determined and there is an inter-observers variation, which is greater than we usually expect. For the problem of (a) Unclear ground truth, (b) limitation of lesional resolution, and (c) diverse OODs, it may be impossible to train an algorithm with non-standardized wide-field photographs. In my perspective, the quality of the image should be at least that of mammogram for accurate diganosis. 

In an onychomycosis study, we detected nail plates in wide-field images, cropped the lesional areas, and annotated them to create a large dataset. Even using the training dataset annotated solely by the algorithm, the algorithm trained with the synthetic onychomycosis dataset showed an expert-level performance in the reader test. The proposed method has the advantage of obtaining a dataset with the same disease prevalence whereas the image dataset in hospitals usually consists of specific diseases according to the dermatologist’s interest.

Model dermatology is a classifier that can detect and classify general skin diseases. For diagnosing using only clinical photographs, the performance of the algorithm was comparable with that of specialists. By using region-based CNN, it is possible to detect nodular lesions and process a large number of photographs without the hard work of an annotator.

CAN5000 is a dataset of 5000 training images for the diagnosis of melanoma and melanocytic nevus. We collected clinical photographs on the Internet. The limitation of this dataset is that it is annotated by the machine based on image findings. The proposed dataset can be used as an additional training dataset along with the private dataset of hospitals. This dataset is not for validation or testing because of the inaccurate ground truth and train-test contamination issue (i.e. internal validation shows meaningless exaggerated results). Algorithms should be validated using the test dataset with clear ground truth in the intended use setting.

## Contributors ##
Han Seung Seog
Soo Ick Cho
