from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import warnings
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import os
warnings.filterwarnings("ignore")
# We only test DenseNet-121 in this script for demo purpose
#from densenet169 import DenseNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# im = cv2.resize(cv2.imread('data/train/dogs/dog.251.jpg'), (224, 224)).astype(np.float32)
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'skirt_length/data/train'
validation_data_dir = 'skirt_length/data/validation'

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

# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(64, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# for layer in base_model.layers:
#     layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

nb_epoch = 1
nb_train_samples = 512
nb_validation_samples = 256

model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        )
# out = model.predict(im)

model.save('models/vgg16_skirt_v2.h5')
