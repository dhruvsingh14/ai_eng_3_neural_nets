##################################
# Week 3.2: Keras Classification #
##################################

# importing libraries
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# for viewing images data
import matplotlib.pyplot as plt

##################
# Importing Data #
##################

# keras library includes mnist dataset as part of api
from keras.datasets import mnist

# reading in mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# checking dimensions
print(X_train.shape)
# interestingly the numbers here represent #images, image_width, image_length


#######################
# Visualizing Image 1 #
#######################
plt.imshow(X_train[0])
plt.show()

# flattening images to 1-D for neural nets to process
num_pixels = X_train.shape[1] * X_train.shape[2] # finds area/size of 1-D vector

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flattening training images 
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flattening test data images

# normalizing pixel values from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255


# dividing target var into categories for better predictability
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

print(num_classes)
# so we have 10 categories


###########################
# Building Neural Network #
###########################
# defining classification model

def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu',input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compiling model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

##############################
# Testing our Neural Network #
##############################

# building the model
model = classification_model()

# fitting the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)

# checking model accuracy
print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))


############################
# Saving Pretrained Models #
############################

model.save('classification_model.h5')

from keras.models import load_model

pretrained_model = load_model('classification_model.h5')



































# in order to display plot within window
# plt.show()
