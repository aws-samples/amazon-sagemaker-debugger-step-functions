import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from model import generate_model
import logging

import smdebug.tensorflow as smd


logger = logging.getLogger(__name__)
logger.setLevel('INFO')

global BATCH_SIZE 
global STARTING_LEARNING_RATE
global WARMUP_LEARNING_RATE 

BATCH_SIZE = 128

def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--warmup_learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    
    parser.add_argument('--add_batch_norm', type=int, choices=[0, 1])
    
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    args, _ = parser.parse_known_args()
    return args


def load_data(loc_dir):
    "channel = 'train' or 'test'"
    
    x = np.load(os.path.join(loc_dir, 'x.npy'))
    y = np.load(os.path.join(loc_dir, 'y.npy'))
    print(f'Loaded channel {loc_dir}, shape {x.shape}.')

    return x, y

if __name__ == "__main__":   

    args = get_args()    
    
    wm_lr            = args.warmup_learning_rate
    lr0              = args.learning_rate
    add_batch_norm   = args.add_batch_norm
    num_epochs       = args.num_epochs
    
    model_dir        = args.model_dir
    train            = args.train
    test             = args.test
    
    logging.info('Input args:')
    for key in args.__dict__.keys():
        print('   {key:<40}: {val}'.format(key=key, val=args.__dict__.get(key)))

    def lr_schedule(epoch):
        """
        Older Keras API. 
        Divide current_lr by 10 after 81 epochs, and again after 122 epochs.
        """
        if epoch < 1:
            return wm_lr  # warmup
        elif epoch < 81:
            return lr0  # warmup
        elif epoch < 122:
            return lr0 / 10
        else:
            return lr0 / 100

    def compile_model(model):
        opt = tf.keras.optimizers.SGD(lr=lr_schedule(0), momentum=0.9)
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
        return model    
    
    x_train, y_train = load_data(args.train)
    x_test, y_test = load_data(args.test)

    
    model = generate_model(add_batch_norm=add_batch_norm)
    model.summary()
    model = compile_model(model)

    callbacks =  [LearningRateScheduler(lr_schedule, verbose=1)]   
    ## allowing to run the script in local model
    if os.path.exists('/opt/ml/input/config/debughookconfig.json'):        
        debugger_hook = smd.KerasHook.create_from_json_file()
        debugger_hook.save_scalar("epoch", num_epochs)
        debugger_hook.save_scalar("batch_size", BATCH_SIZE)
        debugger_hook.save_scalar("train_steps_per_epoch", len(x_train)/BATCH_SIZE)
        debugger_hook.save_scalar("valid_steps_per_epoch", len(x_test)/BATCH_SIZE)    
        callbacks.append(debugger_hook)
    
    model.fit(x_train, y_train, 
              batch_size=BATCH_SIZE, 
              epochs = num_epochs, 
              validation_data=(x_test, y_test), 
              shuffle=True, 
              callbacks=callbacks)
    
    scores = model.evaluate(x_test, y_test, verbose=1)
    
    print('Final scores ...')
    print('   Test loss: ', scores[0])
    print('   Test accuracy: ', scores[1])
    
    model_local_dir = os.path.join(os.environ.get('SM_MODEL_DIR'), '1')
    model.save(model_local_dir)
    print('Saved model at ', model_local_dir)