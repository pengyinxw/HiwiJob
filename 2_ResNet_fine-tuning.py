import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"        # only use gpu0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras

batch_size = 32
epochs = 20
num_classes = 5


x_train = np.load('cifarA_x.npy')
y_train = np.load('cifarA_y.npy')

x_train = x_train.astype('float32')
x_train = preprocess_input(x_train)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)


# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True,
    validation_split = 0.2)
# Generator that generate batches of augmented image data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32, subset="training")
validation_generator = train_datagen.flow(x_train, y_train, batch_size=32, subset="validation")


# ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
print(model.summary())


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False
    
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

# train the model on the new data for a few epochs
model.fit_generator(train_generator, epochs = epochs, steps_per_epoch=len(train_generator), verbose = 2,
                    validation_data=validation_generator, validation_steps=len(validation_generator))

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(model.layers):
   print(i, layer.name)

# we chose to train the top 3 resnet blocks, i.e. we will freeze
# the first 143 layers and unfreeze the rest:
for layer in model.layers[:143]:
   layer.trainable = False
for layer in model.layers[143:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])

print(model.summary())

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(train_generator, epochs = epochs, steps_per_epoch=len(train_generator), verbose = 2,
                    validation_data=validation_generator, validation_steps=len(validation_generator))


#model.save('ResNet50.h5')