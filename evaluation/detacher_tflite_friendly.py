import os, gc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from glob import glob
import yaml

import tensorflow as tf
import numpy as np

from dataset import Dataset
from model import NetworkConstructor
import consts, utils
from losses import Losses


models_dir = consts.Paths.selected_models_dir
model_path = utils.list_select_models(models_dir)

config_name = 'train_config.yml'
with open(os.path.join(model_path, config_name), 'r') as yaml_file:
    tconf = yaml.load(yaml_file, Loader=yaml.FullLoader)
    
tconf['parent_model_path'] = model_path
parent_tconf = tconf.copy()
    
raw_input_shape = tconf['network_input_size']
net_input_shapes = tconf['input_shapes']


def construct_single_model():
    
    netc = []
    single_models = []
    
    n_single_models = len(net_input_shapes)
    
#     for i in range(1):
    for i in range(n_single_models):

        tf.keras.backend.clear_session()
               
        netc = NetworkConstructor(
            input_shape=raw_input_shape,
            single_input_shapes=net_input_shapes,
            model_prefix=tconf['model_prefix'],
            name_prefix='')
        
        ### preprocess part ###
        for layer in tconf['network']['preprocess']:
            layer = utils.keys_validator(layer)
            netc.layer_case_decider(layer)
            
        netc.set_name_prefix(tconf['model_prefix'] % i + 'x_')
        netc.set_curennt_model_idx(i)
        
        for layer in tconf['network']['core']:
            
            layer = utils.keys_validator(layer)
            
            if layer['type'] == 'random_crop':
                layer['tflite'] = True
            
            elif layer['type'] in ['resizing', 'resize']:
                layer['interpolation'] = ['bilinear' for _ in layer['interpolation']]
                utils.printd(layer)
            
            if layer['single_output']:
                layer['single_output'] = False
                layer['output'] = True
            
            netc.layer_case_decider(layer)
        
        
        if 'ensemble' in tconf['network'].keys():
        
            for j, layer in enumerate(tconf['network']['ensemble']):
                layer = utils.keys_validator(layer)

                if layer['type'] in ['merger', 'model_drop', ]:
                    break

                # if 'scaler' in layer['type']:
                if 'compress' in layer['type']:
                    layer['output'] = True
                    netc.layer_case_decider(layer)
                    break

                netc.layer_case_decider(layer)
    
        netc.create_model(show_summary=True, model_name=f'SingleModel_{net_input_shapes[i]}')
        single_models.append(netc.get_model())
    
    return single_models


def construct_ensemble_head():
    
    single_outputs = []
    for j, layer in enumerate(tconf['network']['ensemble']):
        layer = utils.keys_validator(layer)

        if layer['type'] in ['merger', 'model_drop']:
            break

        for i in range(n_single_models):
            netc.set_name_prefix(tconf['model_prefix'] % i + 'x_')
            netc.set_curennt_model_idx(i)
            netc.set_x_single_output(i)
            netc.layer_case_decider(layer)
            single_outputs.append(netc.get_x())

    netc.set_x(single_outputs)
    for layer in tconf['network']['ensemble'][j:]:
        netc.set_name_prefix("EnsembleHead_")
        layer = utils.keys_validator(layer)
        netc.layer_case_decider(layer)


single_networks = construct_single_model()

ensemble = tf.keras.models.load_model(model_path)
optimizer = tf.keras.optimizers.Adam()
ensemble.compile(optimizer, metrics=['accuracy'])
ensemble.summary(160, [.45, .58, .65, 1])

dataset = Dataset(tconf['dataset'])
# batch_size = tconf['train'][0]['batch'] 
batch_size = 32
tfd_test = dataset.get_tfds_test(batch_size, max_n=-1)

for i, network in enumerate(single_networks):
    for layer in network.layers:
        try:
            w_in_ensemble = ensemble.get_layer(layer.name).get_weights()
            layer.set_weights(w_in_ensemble)
            np.testing.assert_equal(w_in_ensemble, layer.get_weights())
        except ValueError as ve:
            continue
#             if 'compress' in str(ve):
#                 continue
#             else:
#                 print(f"Value Error: {ve}")
#                 raise
            
    network.compile(optimizer, metrics=['accuracy'])
    
    
    ### Save model ###
    model_name = tconf['name_prefix'].replace('ensemble_', '') + f'_tflite_friendly_{net_input_shapes[i]}'
#     model_name = '_'.join(tconf['name_prefix'].split('_')[1:]) + f'_tflite_friendly_{net_input_shapes[i]}'
    save_path = consts.Paths.detached_models_dir / model_name
    print(f"Saving model {model_name}...")
    network.save(str(save_path))
    
    # save parent config file
    with open(str(save_path / str('parent_' + config_name)), 'w') as yaml_file:
        yaml.dump(parent_tconf, yaml_file, default_flow_style=False)
        
    # save plot
    tf.keras.utils.plot_model(network, to_file=str(save_path / 'plot.png'),
                              show_shapes=True, expand_nested=True)
        
    ### Evaluation ###
#     evs = network.evaluate(tfd_test, verbose=0)
#     utils.printd(f'Evaluation {network.name}: {evs}')

# evs = ensemble.evaluate(tfd_test, verbose=0)
# utils.printd(f'Evaluation {ensemble.name}: {evs}')
print("Done.")