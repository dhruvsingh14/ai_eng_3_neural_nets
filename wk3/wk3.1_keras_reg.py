##########################################
# Week 3.1: Regression Models with Keras #
##########################################

# importing libraries
import pandas as pd
import numpy as np

##################
# Importing Data #
##################

# downloading data
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()

##############
# Tabulating #
##############

# checking data dimensions
concrete_data.shape

# summarizing data spread / 5 metric
concrete_data.describe()

# checking for null values
concrete_data.isnull().sum()

########################################
# Splitting predictors and Target vars #
########################################

concrete_data_columns = concrete_data.columns

# subsetting to all columns besides strength
predictors = concrete_data[concrete_data_columns[
                            concrete_data_columns != 'Strength']]

# selecting strenght to target var
target = concrete_data['Strength']

# checking our subsetted datasets
print(predictors.head())
print(target.head())

# normalizing data by subtracting mean, and dividing by std dev
predictors_norm = (predictors - predictors.mean()) / predictors.std()
print(predictors_norm.head())

# grabbing number of columns into a variable
n_cols = predictors_norm.shape[1]
print(n_cols)

#######################
# Importing Libraries #
#######################

# tensor flow backend installed to keras
import keras

from keras.models import Sequential
from keras.layers import Dense

#############################
# Building a Neural Network #
#############################

# defining our regression model
def regression_model():
    # creating a model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compiling model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#######################################
# Testing and Training the Neural Net #
#######################################
# building the model by first declaring a regression object
model = regression_model()

# fitting the model to our data
model.fit(predictors_norm, target, validation_split=0.3,epochs=100, verbose=2)

































# in order to display plot within window
# plt.show()
