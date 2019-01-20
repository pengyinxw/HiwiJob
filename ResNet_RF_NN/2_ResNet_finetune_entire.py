import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"        # only use gpu0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
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
    horizontal_flip=True,
    validation_split = 0.2)
# Generator that generate batches of augmented image data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32, subset="training")
validation_generator = train_datagen.flow(x_train, y_train, batch_size=32, subset="validation")


# ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = Dense(512, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)

model = Model(inputs=base_model.input, outputs=outputs)


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

# train the model on the new data for a few epochs
#train_history_1 = model.fit(x_train, y_train, validation_split = 0.2, epochs = epochs, batch_size = batch_size)
train_history_1 = model.fit_generator(train_generator, epochs = epochs, steps_per_epoch=len(train_generator),
                    validation_data=validation_generator, validation_steps=len(validation_generator))


import matplotlib.pyplot as plt
def plot_train_history(train_history, acc, val_acc, loss, val_loss, ACC, LOSS):
    plt.figure()
    plt.plot(train_history.history[acc])
    plt.plot(train_history.history[val_acc])
    plt.title('Training and validation accuracy')
    plt.ylabel(acc)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig(ACC)

    plt.figure()
    plt.plot(train_history.history[loss])
    plt.plot(train_history.history[val_loss])
    plt.title('Training and validation loss')
    plt.ylabel(loss)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig(LOSS)

plot_train_history(train_history_1, 'acc', 'val_acc', 'loss', 'val_loss', 'acc_1.pdf', 'loss_1.pdf')


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(model.layers):
   print(i, layer.name)
'''
# we chose to train the top 3 resnet blocks, i.e. we will freeze
# the first 143 layers and unfreeze the rest:
for layer in model.layers[:7]:
   layer.trainable = False
'''
for layer in model.layers[0:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())


# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

train_history_2 = model.fit_generator(train_generator, epochs = 50, steps_per_epoch=len(train_generator),
                    validation_data=validation_generator, validation_steps=len(validation_generator))

plot_train_history(train_history_2, 'acc', 'val_acc', 'loss', 'val_loss', 'acc_2.pdf', 'loss_2.pdf')

# model.fit(x_train, y_train, validation_split = 0.2, epochs = epochs, batch_size = batch_size, verbose = 2)
model.save('ResNet50.h5')
