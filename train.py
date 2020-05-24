import os
import sys

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from data_generator import DataGenerator
from model import get_model

num_epochs  = 30
num_workers = 8
batch_size  = 8

def train(dataset_path):
    model = get_model()

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

    train_generator = DataGenerator(directory=f'{dataset_path}/train', 
                                    batch_size=batch_size, 
                                    data_augmentation=True)

    val_generator = DataGenerator(directory=f'{dataset_path}/val',
                                batch_size=batch_size, 
                                data_augmentation=False)

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