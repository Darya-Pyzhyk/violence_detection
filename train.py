import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, Dropout, Multiply, Lambda
from tensorflow.keras.optimizers import SGD

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

import os
from data_generator import DataGenerator

import sys

# extract the rgb images 
def get_rgb(input_x):
    rgb = input_x[...,:3]
    return rgb

# extract the optical flows
def get_opt(input_x):
    opt= input_x[...,3:5]
    return opt  

def build_model():

    inputs = Input(shape=(64,224,224,5))

    rgb = Lambda(get_rgb,output_shape=None)(inputs)
    opt = Lambda(get_opt,output_shape=None)(inputs)

    ##################################################### RGB channel
    rgb = Conv3D(
        16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(
        16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(
        32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    rgb = Conv3D(
        32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = Conv3D(
        32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
    rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

    ##################################################### Optical Flow channel
    opt = Conv3D(
        16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = Conv3D(
        16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)

    opt = Conv3D(
        16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = Conv3D(
        16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)

    opt = Conv3D(
        32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = Conv3D(
        32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)

    opt = Conv3D(
        32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(opt)
    opt = Conv3D(
        32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(opt)
    opt = MaxPooling3D(pool_size=(1,2,2))(opt)


    ##################################################### Fusion and Pooling
    x = Multiply()([rgb,opt])
    x = MaxPooling3D(pool_size=(8,1,1))(x)

    ##################################################### Merging Block
    x = Conv3D(
        64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(
        64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(
        64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(
        64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(
        128, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = Conv3D(
        128, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2,3,3))(x)

    ##################################################### FC Layers
    x = Flatten()(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)

    # Build the model
    pred = Dense(2, activation='softmax')(x)
    return Model(inputs=inputs, outputs=pred)

INIT_LR = 0.01
num_epochs  = 30
num_workers = 8
batch_size  = 8

def train(dataset_path):
    model = build_model()

    def scheduler(epoch):
        if epoch % 10 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.7)
        return K.get_value(model.optimizer.lr)
    
    lr_callback = LearningRateScheduler(scheduler)

    checkpoint_path = "training/cp.ckpt"

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path, verbose=1,period=5)

    callbacks = [cp_callback, lr_callback]

    sgd = SGD(lr=INIT_LR, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    train_generator = DataGenerator(directory=f'{dataset_path}/train', 
                                    batch_size=batch_size, 
                                    data_augmentation=True)

    val_generator = DataGenerator(directory=f'{dataset_path}/val',
                                batch_size=batch_size, 
                                data_augmentation=False)
    val_generator = tf.data.Dataset.from_generator(val_generator,(tf.int64, tf.int64, tf.int64, tf.int64, tf.int64))

    model.fit(train_generator, 
    validation_data = val_generator,
    epochs=num_epochs, callbacks=callbacks, 
    verbose=2, max_queue_size=4, 
    steps_per_epoch=len(train_generator),validation_steps=len(val_generator))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print ('Usage: train.py <data_path>')
    else:
        path = sys.argv[1]
        train(path)