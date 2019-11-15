# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples_mine/class_visualization.png "Visualization"
[image11]: ./examples_mine/modified_class_visualization.png "Visualization updated"

[image2]: ./examples_mine/sign_1.png "Before"
[image3]: ./examples_mine/sign_2.png "After"
[image4]: ./examples_mine/signs_all.png "Original image"
[image5]: ./examples_mine/signs_resized.png "Resized and greyscale"
[image6]: ./examples_mine/placeholder.png "Traffic Sign 3"
[image7]: ./examples_mine/placeholder.png "Traffic Sign 4"
[image8]: ./examples_mine/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is  (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how each of the classes is represented in the training data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step I decided to convert images to gray scale. This step reduces the size of the network.

As the next step I generated additional data. I did this because, as seen from previous image, the training set is highly unbalanced.
To generate a new data I added Gaussian blur with the probability 50% and then add random rotation (in range +/-25 deg), random scaling (in range 0.9 -1) and random translation (in range +/-5 px). I was applying this transformation to random object of a underrepresented  type until there are at least 1000 images of each type.

As a final step I normalized the image data to help the optimizer.

Here is an example of an original image and an augmented image:

![alt text][image2]
![alt text][image3]


Here is a visualization of the new dataset:

![alt text][image11]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was based on LeNet structure. I only added two additional dropout layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding,  Output = 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  output 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding,  Output = 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  output 5x5x16 				|
| Flatten | output 400 |
| Fully connected		|Output  120      									|
|			RELU			|												|
|			Dropout			|						probability=0.9						|
| Fully connected		|Output  84      									|
|			RELU			|												|
|			Dropout			|						probability=0.9					|
| Fully connected		|Output  42 (num_classes)   									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an batch size 256 and learning rate 0.001. I was training 100 epochs. I was playing around with this values, but further tuning them might still lead to slightly better overall network performance. 

I did not change the mu and sigma parameters.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


My base solution was to use the unchanged LeNet architecture. It was relatively easy object recognition task, so I decided that there is no need to use more complex architecture.

 At the end I decided to add a droput layers to each hidden fully connected layer to prevent overfitting.

With the data augmentation in the dataset, I was able to fulfill the project requirements, so I did not introduce any other changes.


My final model results were:
* training set accuracy of 0.9948732830357749
* validation set accuracy of 0.9433106578937193
* test set accuracy of 0.9231195567528218

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. Left image shows the original images and the right image shows the resized images in grayscale, that was feed to the network.

![alt text][image4] ![alt text][image5]

[ True  True  True False  True]

True labels: [40 14 23 37 38]

Detected labels: [40 14 23 18 38]

Potential detection difficulties :
  1. In the first image the perspective might be a bit unusual, also it has a reflection.
  2. On the fourth image the object of interest does not occupy the whole space
  4. The proportions of the fourth image are a bit distorted.
  5. Fourth image should not create any problems.
  3. On the last image the sign is not in the center and do not occupy a lot of space.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Roundabout mandatory					|  	Roundabout mandatory							|
| Stop	      		| Stop 				 				|
| Slippery Road			| Slippery Road      							|
| Go straight or left		| General caution    				|
| Keep right					|  	Keep right							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This worse performance on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


      TopKV2(values=array([[1.00000000e+00, 1.74280573e-11, 1.14693172e-12, 4.22941982e-14,
      7.38172405e-17],
      [9.98980105e-01, 9.12576215e-04, 1.01483936e-04, 3.61861248e-06,
      2.16449280e-06],
      [1.00000000e+00, 1.89664107e-09, 1.72933230e-11, 1.21143828e-12,
      4.24687439e-13],
      [9.98074770e-01, 1.08045759e-03, 7.61427451e-04, 7.91893253e-05,
      3.76461844e-06],
      [9.99755204e-01, 2.44769442e-04, 7.54222518e-10, 9.67873091e-12,
      7.71704065e-20]], dtype=float32), indices=array([[40, 34, 16, 14, 37],
      [14, 17, 38, 34, 23],
      [23, 11, 20, 30, 28],
      [18, 40, 37, 29, 25],
      [38, 34, 36, 17, 30]], dtype=int32))

All the predictions, even the incorrect one, have a very high probabilities. In case of a wrongly predicted sign (row number 4), the correct prediction has third highest probability (7.61427451e-04).


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
