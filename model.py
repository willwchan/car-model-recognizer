import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.python import keras
from tensorflow.python.keras.applications import ResNet152
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

#number of files in car_data/train
num_classes = 196

#weights taken from https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6
resnet_weights_path = 'resnet152_weights_tf.h5'

my_model = Sequential()
my_model.add(ResNet152(include_top=False, pooling='avg', weights=resnet_weights_path))
my_model.add(Dense(num_classes, activation='softmax'))

#code for not training first layer (aka the resnet model)
my_model.layers[0].trainable = False

my_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

image_size = 300
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory()
