Breast Cancer Detection Using Convolutional Neural Networks


Abstract
    Breast cancer is one of the leading causes of death by cancer for women. The automatic detection of breast cancer cells by analyzing histopathological images plays a significant role for patients and their prognosis. In this project, we aim to compare 3 different pre-trained CNN models which are Xception, Visual Geometry Group-16 (VGG-16) and Visual Geometry Group-19 (VGG19),  to build a better understanding of how to choose model when we deal with cancer detecting task. By comparing the F1 score, accuracy and area under receiver operating characteristic curve of  these 3 models, we conclude that VGG19 performs better in breast cancer detection.
.
Introduction 
    Breast cancer is one of the most common and dangerous cancers impacting women worldwide. The histopathological breast cancer image classification greatly impact the diagnosis and treatment of breast cancer in real world medical context. Recently, researchers have paid more efforts into the research of automatically detection and classification of breast cancer using machine learning and computer vision models, in order to improve the diagnosis accuracy and release the pressures of doctors by automate the diagnosis procedures. Spanhol, F. et al.[1] create a high-resolution breast cancer histopathological image dataset for research purpose, Selvathi, D et al.[2], Spanhol, F. et al.[3] both work on the classification task on the breast cancer histopathological image dataset and reach accuracy about 80%.
   Convolutional Neural Networks(CNNs) are widely used in computer vision tasks in recent years. Its ability to automatically extract features and its superior performance on computer vision tasks like segmentation, detection and classification makes it also a popular model in many vision tasks of medical domain.  Causey, Jason L. et al.[6] and Ozdemir, Onur et al.[7] use different approaches with different CNNs architectures designed by themselves to solve the detection and diagnosis problems of lung cancer with CT scans. Spanhol, F et al.[5] use CNNs for feature extraction in breast cancer classification task and split the classification task into feature extraction and classification using different models.  

Dataset summary 
    The Breast Cancer Histopathological Image Classification (BreakHis)[1] is  composed of 9,109 microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X, 200X, and 400X). It contains 2,480  benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit depth in each channel, PNG format). 
    We only use 2 classes (Benign and Malignant) and samples from all magnification factors (40X, 100X, 200X, and 400X) for this project. And we evaluate the performance of models on the binary classification problem.

Hypothesis 
1.  Using different pretrain CNNs networks to extract features have different classification performance on the BreakHis dataset, and some pretrained models are more suitable for the specific breast cancer histopathological classification task. Histopathological tasks share some common properties which make our model generalized for other medical histopathological tasks.
2.   When using the same model for classification, histopathological images with different magnification factors (40X, 100X, 200X, and 400X) will have different classification accuracy and different F1 prediction score, images with higher resolution contains more information and will give a better classification performance if our model is good enough to extract all useful information. 


Experimental Design and Methods
1.Image Pre-processing.
Gaussian Filter
In electronic and signal processing, gaussian noise is often used to simulate the worst case of noise that affect the result of electronic devices. We use gaussian noises to simulate the noises of our histopathological images, and it is frequent for noises like Johnson noise to appear in images taken by electronic cameras. In frequency domain, Gaussian filter is a gaussian distribution function, and its simplest form can be performed using a mask in time domain. We use a gaussian filter with kernel size equal to 3.  
RGB to HSV
HSV stands for hue, saturation and value. It is an alternative color space representation to RGB space. In image processing, researchers often use HSV or HSL to represent the image instead of RGB space, and RGB space is more suitable for screen display, because first, the three channels of RGB are all strongly related to brightness, and secondly, to most people, the red color is less sensible than the blue color, so if we use distance like the Euclidean Distance to measure the distance between three RGB channels, the result will be largely different from the intuitive vision of our eyes. We use both RGB and HSV space as our input, and concatenate the RGB channel and HSV channel together before input into the feature extraction model.
Image Augmentation
Image augmentation artificially creates training images through different ways of processing or combination of multiple processing, such as random rotation, shifts, shear and flips, etc. Deep neural network is highly data-driven and the deeper the network, the easier we overfit the data, which makes data augmentation important for our relatively small dataset when we introduce deep neural network as our feature extractor. Also, data augmentation adds more variance to the dataset, which makes the model we trained more robust to the changes of real world context. For each original image, we only generate two images for RGB space and two images for HSV channel, considering the limited space of the disk provided by Colab and time consuming.

2. Feature extraction.
After pre-processing, each image was then resized to meet the requirement of certain pre-trained CNN model. Afterward, we implemented 3 pre-trained models to extract the features from the images:
Xception

Xception is made up with depth-wise separable convolution which is a pointwise convolution followed by a depthwise convolution. Xception takes the Inception hypothesis to the eXtreme. Firstly, cross-channel (or cross-feature map) correlations are captured by 1x1 convolutions. Consequently, spatial correlations within each channel are captured via the regular 3x3 or 5x5 convolutions. Taking the above indea to an extreme means performing 1x1 to every channel, then performing a 3x3 to each output, which is identical to replacing the Inception module with depth wise separable convolutions. Xception takes image of 299x299 as input.

VGG-16 has 13 convolutional and 3 fully-connected layers, carrying with them the ReLU tradition from AlexNet. This network stacks more layers onto AlexNet, and use smaller size filters (2×2 and 3×3). It consists of 138M parameters and takes up about 500MB of storage space. The network has an image input size of 224x224.

VGG-19 is a deeper variant of VGG-16 with 3 more convolutional layers added, thus 19 layers deep in total. Similar to VGG-16, VGG-19 also has an image input size of 224x224.

After feature extraction, features of all images from given magnification factor were stored in a 4D numpy array file like (N, rows, columns, channels) where N is the number of images. Hence we have 4 such files for each model, 12 files in total, which will then be fed into the classifier to make the classification.

3. Classification
The data set is shuffled and split into 80% training set and 20% testing set. In both
training and testing sets, there is 66-69% positive samples in each different magnification subset. A logistic regression classifier is implemented for classification.

F1 score, Area under the ROC curve and accuracy which measures the average accuracy among patients are calculated for each subset to evaluate the final result.
The ROC curves and detailed measures are shown in the code link.

Results

Our result shows that VGG19 performs better than Xception and VGG16 in all three metrics for images of 100X, 200X, 400X, while for images of 40X the Xception model slightly outperforms VGG-19. Comparing the two VGG-16 models, VGG-19 always achieves high accuracy. It  demonstrates that the representation depth is beneficial for the classification accuracy since the biggest difference between VGG-16 and VGG-19 is the depth. The VGG19 consists of 19 layers of deep neural network whereas the VGG16 consists of 16 layers respectively.

Comparing different magnification factor groups, the smaller the magnification factor is, the higher accuracy it achieves. This can be explained by using smaller magnification is good for showing the whole picture so that models will be better at capturing useful information from the slice. 



References
Provide any citations and/or links to notebooks, datasets, etc
[1] Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L., A Dataset for Breast Cancer Histopathological Image Classification, IEEE Transactions on Biomedical Engineering (TBME), 63(7):1455-1462, 2016.

[2] Selvathi, D., & Poornila, A. (2018). Deep Learning Techniques for Breast Cancer Detection Using Medical Image Analysis. In (pp. 159-186).
 
[3] Spanhol, F., Oliveira, L. S., Petitjean, C., and Heutte, L., Breast Cancer Histopathological Image Classification using Convolutional Neural Network, International Joint Conference on Neural Networks (IJCNN 2016), Vancouver, Canada, 2016.

[4] Xie, J., Liu, R., Luttrell, J. t., & Zhang, C. (2019). Deep Learning Based Analysis of Histopathological Images of Breast Cancer. Frontiers in genetics, 10, 80-80. doi:10.3389/fgene.2019.00080
 
[5] Spanhol, F., Cavalin, P.,  Oliveira, L. S., Petitjean, C., Heutte, L., Deep Features for Breast Cancer Histopathological Image Classification, 2017 IEEE International Conference on Systems, Man, and Cybernetics (IEEE SMC 2017), Banff, Canada, 2017

[6] Causey, Jason L.; Guan, Yuanfang; Dong, Wei; Walker, Karl; Qualls, Jake A.; Prior, Fred; Huang, Xiuzhen, Lung cancer screening with low-dose CT scans using a deep learning approach, arXiv:1906.00240, 2019

[7] Ozdemir, Onur; Russell, Rebecca L.; Berlin, Andrew A., A 3D Probabilistic Deep Learning System for Detection and Diagnosis of Lung Cancer Using Low-Dose CT Scans, arXiv:1902.03233, 2019


