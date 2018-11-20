import csv
import cv2
import numpy as np
import os

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#creates a test set, using 5% of the data
from sklearn.model_selection import train_test_split
train_samples, test_samples = train_test_split(samples, test_size=0.05)

#creates train and validation sets, using 80/20 split of the data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn
from numpy.random import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            #This section reads each sample from the batch and gathers the data from the appropriate columns
            for line in batch_samples:

                #this collects the images, using their paths
                center_path = 'data/IMG/' + line[0].split('\\')[-1] 
                left_path = 'data/IMG/' + line[1].split('\\')[-1]
                right_path = 'data/IMG/' + line[2].split('\\')[-1]
                image_center = cv2.imread(center_path) 
                image_left = cv2.imread(left_path)
                image_right = cv2.imread(right_path)
                images.extend([image_center,image_left, image_right])
    
                #this adds the measurements and necessary corrections
                correction = 0.17
                measurement_center = float(line[3])	
                measurement_left = measurement_center + correction    
                measurement_right = measurement_center - correction   
                measurements.extend([measurement_center, measurement_left, measurement_right])

            #this section creates a symmetrical duplicate of the data, to get a broader dataset
            augmented_images, augmented_measurements = [], []
            for image,measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
test_generator = generator(test_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D

drop_rate = 0.2

#Model architecture

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(3,160,320)))
#shape: 3@60x320 
model.add(Dropout(drop_rate))
model.add(Convolution2D(24,4,5,subsample=(2,3),activation="relu"))
#shape: 24@29x106
model.add(Dropout(drop_rate))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
#shape: 36@13x51
model.add(Dropout(drop_rate))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
#shape: 48@5x24
model.add(Dropout(drop_rate))
model.add(Convolution2D(64,3,3,activation="relu"))
#shape: 64@3x22
model.add(Dropout(drop_rate))
model.add(Convolution2D(64,3,3,activation="relu"))
#shape: 64@1x20
model.add(Flatten())
model.add(Dropout(drop_rate))
model.add(Dense(1280,activation="relu"))
model.add(Dropout(drop_rate))
model.add(Dense(100,activation="relu"))
model.add(Dropout(drop_rate))
model.add(Dense(50,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))

#model training
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*3*2, \
		validation_data=validation_generator, \
            	nb_val_samples=len(validation_samples)*3*2, nb_epoch=3)

model.save('model.h5')

#evaluate model
print("Testing")

test_loss = model.evaluate_generator(test_generator, val_samples = len(test_samples)*3*2)

print("")
print("Test Loss:")
print(test_loss)

