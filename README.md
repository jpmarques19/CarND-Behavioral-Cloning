# Behavioral Cloning 


---

Behavioral Cloning Project

The goals  steps of this project are the following
 Use the simulator to collect data of good driving behavior
 Build, a convolution neural network in Keras that predicts steering angles from images
 Train and validate the model with a training and validation set
 Test that the model successfully drives around track one without leaving the road
 Summarize the results with a written report

[//]: # (Image References)

[image1]: ./nvidia.jpg "Model Visualization"
[image2]: ./center.jpg "Center Image"
[image3]: ./left.jpg "Left Image"
[image4]: ./right.jpg "Right Image"
[image5]: ./training.jpg "Training results"

## Rubric Points
### Here I will consider the [rubric points](httpsreview.udacity.com#!rubrics432view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files
 model.py containing the script to create and train the model
 drive.py for driving the car in autonomous mode
 model.h5 containing a trained convolution neural network 
 writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 64-78) 

The model includes RELU layers to introduce nonlinearity (code line 66), and the data is normalized in the model using a Keras lambda layer (code line 62). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (see code line 65). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 92). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I did a total of 4 laps on track 1. Two laps forward and two laps in reverse direction. I kept the car mostly on the center of lane.


For details about how I created the training data, see the next section. 

### Architecture and Training Documentation
#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one presented in [this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) published by NVIDIA.

In order to gauge how well the model was working, I split my image and steering angle data into a training, validation and test set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include dropout layers throughout the model.
I first used a drop rate of 0.5 but the model was performing baddly on the track. I eventually used a drop rate of 0.2 and that was enough to keep the model from overfitting while mantaining the car on the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 65-92) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					                | 
|:---------------------:|:-------------------------------------------------------------:| 
| Input         		| 160x320x1 color image   					                    | 
| Normalization      	|                                                            	|
| Cropping				| outputs 3@60x320								                |
| Convolution 4x5	    | 2x3 stride, valid padding, relu activation, outputs 24@29x106	|
| Convolution 5x5	    | 2x2 stride, valid padding, relu activation, outputs 36@13x51	|
| Convolution 5x5     	| 2x2 stride, valid padding, relu activation, outputs 48@5x24 	|
| Convolution 3x3	    | 1x1 stride, valid padding, relu activation, outputs 64@3x22  	|
| Convolution 3x3	    | 1x1 stride, valid padding, relu activation, outputs 64@1x20  	|
| Flatten        		| 		                                                   		|
| Fully connected		| inputs 1280, relu activation                     				|
| Fully connected		| inputs 100 , relu activation      		            		|
| Fully connected		| inputs 50 ,  relu activation         			            	|
| Fully connected		| inputs 10 ,  relu activation         			            	|
| Output                | Scalar (Steering angle)        			            	|


Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. After that I recorded another two laps but in reverse direction. Here is an example image of center lane driving:

![alt text][image2]

I used the left and right cameras to teach the car how to make recoveries from the sides of the road.

![alt text][image3]
![alt text][image4]

To augment the data set, I also flipped the images and angles.

After the collection process, I had over 20000 images. I then preprocessed this data by normalizing and cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by some experimentation. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image5]

We can see that the test_loss is aligned with the validation_loss tendency in training.