import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, Dropout, Multiply, Lambda
from tensorflow.keras.optimizers import SGD

model_path = 'model/model.h5'

def get_model(load = True):
    if load:
        model = tf.keras.models.load_model(model_path)
    else:
        model = _build_model()
        
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# extract the rgb images 
def _get_rgb(input_x):
    rgb = input_x[...,:3]
    return rgb

# extract the optical flows
def _get_opt(input_x):
    opt= input_x[...,3:5]
    return opt  

FRAMES_PER_VIDEO = 64

def _build_model():

    inputs = Input(shape=(FRAMES_PER_VIDEO,224,224,5))

    rgb = Lambda(_get_rgb,output_shape=None)(inputs)
    opt = Lambda(_get_opt,output_shape=None)(inputs)

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

