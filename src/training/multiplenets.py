import os, gc
from pathlib import Path
import shutil
from datetime import datetime

import yaml

code_path = Path(os.path.realpath(__file__))
dir_path = code_path.parent

config_name = 'train_config.yml'
with open(dir_path / config_name, 'r') as yaml_file:
    tconf = yaml.load(yaml_file, Loader=yaml.FullLoader)

import tensorflow as tf

from dataset import Dataset
from model import NetworkConstructor
import consts, utils
from losses import Losses

# Avoid fully occupation of gpu memory
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

    
# ==================================================================================== #


input_shapes = tconf['input_shapes']

shapes_str = ','.join([str(shape) for shape in input_shapes])

if tconf['name_prefix'] is None:
    tconf['name_prefix'] = 'ensemble'
model_name = f"{tconf['name_prefix']}_({shapes_str})_%s"

if tconf['model_name'] is None:
    tconf['model_name'] = datetime.now().strftime('%y%m%d_%H%M%S')
model_name = model_name % tconf['model_name']
utils.printd(f"MODEL NAME: {model_name}")

model_path = consts.Paths.trained_models_dir / model_name
input_size = tconf['network_input_size']
# input_size = [int(s) for s in tconf['network_input_size'].split(',')]


    
def construct_model():
    
    netc = NetworkConstructor(
        input_shape=input_size,
        single_input_shapes=input_shapes,
        model_prefix=tconf['model_prefix'],
        name_prefix=''
    )
    
    ### preprocess part ###
    for layer in tconf['network']['preprocess']:
        layer = utils.keys_validator(layer)
        netc.layer_case_decider(layer)
    
    ### single models part ###
    n_single_models = len(input_shapes)
    preped_x = netc.get_x()
    for i in range(n_single_models):
        
        netc.set_x(preped_x)
        netc.set_name_prefix(tconf['model_prefix'] % i + 'x_')
        netc.set_curennt_model_idx(i)
        
        for layer in tconf['network']['core']:
            layer = utils.keys_validator(layer)
            netc.layer_case_decider(layer)
            
    ### ensemble part ###
    if 'ensemble' in tconf['network'].keys():
        single_outputs = []
        for i in range(n_single_models):
            netc.set_name_prefix(tconf['model_prefix'] % i + 'x_')
            netc.set_curennt_model_idx(i)
            netc.set_x_single_output(i)
            
            for j, layer in enumerate(tconf['network']['ensemble']):
                
                if layer['type'] in ['merger', 'model_drop']:
                    break
                
                layer = utils.keys_validator(layer)
                netc.layer_case_decider(layer)
                
            
            single_outputs.append(netc.get_x())
        
        netc.set_x(single_outputs)
        
        for layer in tconf['network']['ensemble'][j:]:
            netc.set_name_prefix("EnsembleHead_")
            layer = utils.keys_validator(layer)
            netc.layer_case_decider(layer)
            
    netc.create_model(show_summary=True)
    
    return netc
            
        
        
        
#         single_outputs = []
#         for j, layer in enumerate(tconf['network']['ensemble']):
#             layer = utils.keys_validator(layer)

#             if layer['type'] in ['merger', 'model_drop']:
#                 break

#             for i in range(n_single_models):
#                 netc.set_name_prefix(tconf['model_prefix'] % i + 'x_')
#                 netc.set_curennt_model_idx(i)
                
# #                 netc.set_x_single_output(i)
                    
#                 netc.layer_case_decider(layer)
#                 single_outputs.append(netc.get_x())

#         netc.set_x(single_outputs)

#         for layer in tconf['network']['ensemble'][j:]:
#             netc.set_name_prefix("EnsembleHead_")
#             layer = utils.keys_validator(layer)
#             netc.layer_case_decider(layer)
    
#     netc.create_model(show_summary=True)
    
#     return netc
                                 

def train(model, train_step_config, single_outputs_names, final_outputs_names):
    
    # Prepare losss
    n_single_outputs = len(single_outputs_names)
    n_final_outputs = len(final_outputs_names)
    loss_coeffs = train_step_config['loss_coeffs']
    
    if isinstance(loss_coeffs, float):
        single_outputs_coeffs = [loss_coeffs] * n_single_outputs
        final_outputs_coeffs = [loss_coeffs] * n_final_outputs # if n_final_outputs > 1 else loss_coeffs
    
    elif isinstance(loss_coeffs, str):
        loss_coeffs = [float(v) for v in loss_coeffs.split()]
        if len(loss_coeffs) == 2:
            single_outputs_coeffs = loss_coeffs[:1] * n_single_outputs
            final_outputs_coeffs = loss_coeffs[1:] * n_final_outputs # if n_final_outputs > 1 else loss_coeffs[1]
        elif len(loss_coeffs) == n_final_outputs + n_single_outputs:
            single_outputs_coeffs = loss_coeffs[:n_single_outputs]
            final_outputs_coeffs = loss_coeffs[n_single_outputs:]
    
    Loss = Losses(single_outputs_coeffs=single_outputs_coeffs,
                  final_outputs_coeffs=final_outputs_coeffs,
                  single_outputs_names=single_outputs_names,
                  final_outputs_names=final_outputs_names)
    
    losses, loss_weights = Loss.get_loss_dict()
    
    
    optimizer = tf.keras.optimizers.Adam(train_step_config['lr'])
    
    
    # Check whether model has ensemble head
    monitor = 'accuracy'
    if len(final_outputs_names) > 1:
        monitor = 'final_prediction_accuracy'
#     monitor = final_outputs_names[-1] + '_accuracy'
#     if final_outputs_names == single_outputs_names:
#         monitor = 'accuracy'
            
    callbacks = get_callbacks(train_step_config['callbacks'].split(), monitor=monitor)
    
    tfd_train = dataset.get_tfds_train(train_step_config['batch'])
    tfd_test = dataset.get_tfds_test(train_step_config['batch'] * 2)
    
    if train_step_config['trainable_idx'] is not None:
        model = model_c.set_trainable_param(model, train_step_config['trainable_idx'])
    
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])
    model.fit(tfd_train, validation_data=tfd_test, epochs=train_step_config['epoch'], callbacks=callbacks, verbose=2)


def get_callbacks(cb_keys, monitor='val_final_prediction_accuracy'):
    logs = str(model_path / 'logs')
    cbs = {
        'RoLRP': tf.keras.callbacks.ReduceLROnPlateau(monitor, factor=.98, patience=0, verbose=1),
        'ES': tf.keras.callbacks.EarlyStopping(monitor, patience=20),
        'MC': tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor, save_best_only=True),
        'TB': tf.keras.callbacks.TensorBoard(log_dir=logs)
    }
    
    return [cbs[key] for key in cb_keys]

                                 
def evaluate(do_save=False):
    tfd_test = dataset.get_tfds_test(tstep['batch'] * 2, max_n=-1)
    evs = model.evaluate(tfd_test, verbose=0)
    msg = ''
    msg += f' {final_outputs_name[-1]:<30}{evs[-1]*100:.2f}%'
    
    # Check whether model has ensemble head
    if final_outputs_name != single_outputs_name:
        for i, name in enumerate(single_outputs_name[::-1]):
            msg += f'\n {name:<30}{evs[-(i+2)]*100:.2f}%'
        
    # Save
    if do_save:
        gc.collect()
        with open(model_path / 'evaluation.txt', 'w') as file:
            file.write(msg)
        gc.collect()
    
    msg = 'Evaluation Step:' + '\n' + msg
    utils.printd(msg, '#')
                                 

def save_everything():
    ## Save configs, code, and model plot
    utils.check_create_path(model_path)
    tf.keras.utils.plot_model(model, to_file=str(model_path / 'plot.png'),
                              show_shapes=True, expand_nested=True)
    shutil.copy(str(code_path), str(model_path / 'training_code.py'))
    shutil.copy(str(dir_path / config_name), str(model_path / config_name))

    

if __name__ == '__main__':
    
    dataset = Dataset(tconf['dataset'])

    model_c = construct_model()
    
    model = model_c.get_model()
    single_outputs_name, final_outputs_name = model_c.get_outputs_names()
    
    save_everything()

    # Training steps
    start_time = datetime.now()
    for tstep in tconf['train']:

        gc.collect()
        utils.printd(model_name, '*')
        utils.printd(tstep)

        train(model, tstep, single_outputs_name, final_outputs_name)

        elapsed = datetime.now()
        utils.printd('Elapsed Time: '+''.join(str((elapsed - start_time)).split('.')[:-1]))

        do_save = 'save' in tstep.keys() and tstep['save']

        # Evaluate
        gc.collect()
        evaluate(do_save)

        # Save model
        if do_save:
            print("Saving Model ...")
            model.save(str(model_path))
