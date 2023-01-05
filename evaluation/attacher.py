import os, gc
from pathlib import Path
from glob import glob
import yaml

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import numpy as np

# Avoid fully occupation of gpu memory
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from dataset import Dataset
from model import NetworkConstructor
import consts, utils
from losses import Losses
import custom_layers as CL


def get_model_input_shape(model, only_height=True):
    
    for i, l in enumerate(model.layers):
        if 'randcrop' in l.name:
            
            if only_height:
                return model.layers[i+1].input_shape[1]
            
            return model.layers[i+1].input_shape[1:3]
    
    layers_config = model.get_config()['layers']
    
    for l in layers_config:
        if l['class_name'] == 'RandomCrop':
            height = l['config']['height']
            
            if only_height:
                return height
            
            width = l['config']['width']
            
            return height, width
        
    raise "No actual input shape was found."
    
            
def get_config(config_path):

    with open(str(config_path), 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        
    return config


def get_size_to_depthwise_multiplier_mapping(config):
    
    for layer in config['network']['core']:
        
        if 'random_crop' in layer['type']:
            sizes = layer['size']
        
        if layer['type'] in ['mobilenetv2']:
            alpha_list = layer['alpha']
    
    return {k:v for k, v in zip(sizes, alpha_list)}


def get_model_real_input_shape(model):
    
    for i, l in enumerate(model.layers):
        if 'crop' in l.name:
            return model.layers[i+1].input_shape[1:3]
    
    config = model.get_config()
    
    for layer in config['layers']:
        if 'crop' in layer['class_name'].lower():
            return layer['config']['height'], layer['config']['width']
        
    raise "Crop is not in layers"
        

def assemble(models, config):
    '''Gets a list of loaded models and ensembles them'''
    
    network_in_size = config['network_input_size']
    net_input_shapes = config['input_shapes']
    
    netc = NetworkConstructor(
        input_shape=network_in_size,
        single_input_shapes=net_input_shapes,
        model_prefix=config['model_prefix'],
        name_prefix=''
    )
    
    layer = {
        'type': 'multiple_models',
        'models': models,
    }
    
    layer = utils.keys_validator(layer)
    netc.layer_case_decider(layer)
    
    network_outputs = netc.get_x()
    
    # Create ensemble head
    if 'ensemble' in config['network'].keys():
        
        ### Singel Model Parts ###
        single_outputs = []
        n_single_models = len(models)
        x_values = netc.get_x()
        size_to_dwm_mapping = get_size_to_depthwise_multiplier_mapping(config)
        utils.printd(f'size_to_dwm_mapping: {size_to_dwm_mapping}')
        dwm = []
        
        for idx in range(n_single_models):
            
            netc.set_name_prefix(config['model_prefix'] % idx + 'x_')
            netc.set_curennt_model_idx(idx)
            
            for j, layer in enumerate(config['network']['ensemble']):
                
                layer = utils.keys_validator(layer)
                
                if layer['type'] in ['merger', 'model_drop']:
                    break
                
                elif layer['type'] == 'output_decompressor':
                    x = x_values[idx]
                    netc.k_in_top_k = tf.shape(x[1])[-1] // 2
                    netc.output_shape = tf.shape(x[0])
                    netc.set_x(x[1])
                    netc.layer_case_decider(layer)
                    
                elif layer['type'] == 'output_scaler':
                    dwm.append(size_to_dwm_mapping[get_model_real_input_shape(models[idx])[0]])
                    netc.depthwise_multipliers = dwm
                    netc.layer_case_decider(layer)
            
            single_outputs.append(netc.get_x())
        
        netc.set_x(single_outputs)
        
        ### Ensemble Head ###
        for layer in config['network']['ensemble'][j:]:
            netc.set_name_prefix("EnsembleHead_")
            layer = utils.keys_validator(layer)
            netc.layer_case_decider(layer)
            
    netc.create_model(show_summary=True)
    
    return netc


if __name__ == '__main__':
    
    models_dir = consts.Paths.detached_models_dir
    model_paths = utils.list_select_models(models_dir)
    
    model_paths = [model_paths] if not isinstance(model_paths, list) else model_paths
    config = get_config(Path(model_paths[0]) / 'parent_train_config.yml')
    
    models = []
    config['input_shapes'] = []
    model_names = []
    for i, mpath in enumerate(model_paths):
        model_names.append(mpath.split(os.sep)[-1])
        print(f"Loading model {model_names[-1]}...")
        
        loaded_model = tf.keras.models.load_model(mpath)
        loaded_model = utils.add_update_prefix(
            model=loaded_model, 
            prefix=f"model{i+1}", 
            old_prefix="model[0-9]+",
            custom_objects={
                'Compression': CL.Compression, 
                'RandomCrop2D': CL.RandomCrop2D, 
                'RandAugment': CL.RandAugment
            })
        
        models.append(loaded_model)
        loaded_model.summary(160, [.45, .58, .65, 1])
        
#         config['input_shapes'].append(loaded_model.input.shape[1])
        config['input_shapes'].append(get_model_input_shape(loaded_model, only_height=True))

    ensemble = assemble(models, config).get_model()
    
    optimizer = tf.keras.optimizers.Adam()
    metrics=[tf.keras.metrics.TopKCategoricalAccuracy(5), 'accuracy']
    ensemble.compile(optimizer, metrics=metrics)
    
    dataset = Dataset(config['dataset'])
    # batch_size = tconf['train'][0]['batch'] 
    batch_size = 256
    tfd_test = dataset.get_tfds_test(batch_size, max_n=-1)
    
    accuracies = []
    top_5_accs = []
    for i in range(20):
        evs = ensemble.evaluate(tfd_test, verbose=0)
        
        accuracies.append(round(evs[2] * 100, 3))
        top_5_accs.append(round(evs[1] * 100, 3))
        utils.printd(f'Evaluation #{i+1}   {ensemble.name}: {accuracies[-1]}\t top5: {top_5_accs[-1]}')

    utils.printd(f'Models: {model_names}')
    
    utils.printd(f'Accuracies: {accuracies}')
    accuracies = np.array(accuracies)
    
    utils.printd(f'top-5: {top_5_accs}')
    top_5_accs = np.array(top_5_accs)
    
    utils.printd(f'Mean Top-5 Accuracy (20 runs): {top_5_accs.mean(): .2f} +/-{top_5_accs.std(): .2f}')
    utils.printd(f'Mean Accuracy (20 runs): {accuracies.mean(): .2f} +/-{accuracies.std(): .2f}')
    
#     model_paths = sorted(glob(str(models_dir / "*")))
#     print(model_paths)
                   
#     netc = NetworkConstructor(
#         input_shape=raw_input_shape,
#         single_input_shapes=net_input_shapes,
#         model_prefix=tconf['m