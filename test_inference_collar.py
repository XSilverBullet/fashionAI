"""Test ImageNet pretrained DenseNet"""
import os
import cv2
import numpy as np
from keras.optimizers import SGD,adam
import keras.backend as K
import warnings
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings("ignore")
# We only test DenseNet-121 in this script for demo purpose
from densenet121 import DenseNet
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import GlobalAveragePooling2D
# im = cv2.resize(cv2.imread('data/train/dogs/dog.251.jpg'), (224, 224)).astype(np.float32)
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'collar_design/train'
validation_data_dir = 'collar_design/validation'

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=16,
            class_mode='categorical')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')
# Subtract mean pixel and multiple by scaling constant
# Reference: https://github.com/shicai/DenseNet-Caffe
# im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
# im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
# im[:,:,2] = (im[:,:,2] - 123.68) * 0.017

if K.image_dim_ordering() == 'th':
  # Transpose image dimensions (Theano uses the channels as the 1st dimension)
  # im = im.transpose((2,0,1))

  # Use pre-trained weights for Theano backend
  weights_path = 'imagenet_models/densenet121_weights_th.h5'
else:
  # Use pre-trained weights for Tensorflow backend
  weights_path = 'imagenet_models/densenet121_weights_tf.h5'

# Insert a new dimension for the batch_size
# im = np.expand_dims(im, axis=0)

# Test pretrained model
model = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path, category_num=5)
# x = model.output
# x = Dense(6, name='fc61')(x)
# x = Activation('softmax', name='prob1')(x)
# model = Model(model.input, x, name='densenet1')
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

nb_epoch = 20
nb_train_samples = 512
nb_validation_samples = 256

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch)
# out = model.predict(im)

model.save_weights('models/densenet121_collar.h5')
print(model.evaluate_generator(validation_generator, 10))
#predict = model.predict_generator
# Load ImageNet classes file
# classes = []
# with open('resources/classes.txt', 'r') as list_:
#     for line in list_:
#         classes.append(line.rstrip('\n'))
#
# print('Prediction: '+str(classes[np.argmax(out)]))
