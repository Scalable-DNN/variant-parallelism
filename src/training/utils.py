import os, tempfile, re
from pathlib import Path
from glob import glob
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import consts


def keys_validator(layer: dict, keys: set=None):
    keys = {'type', 'interpolation', 'new_size', 'alpha', 'include_top', 'softmax'
            'input_shapes', 'trainable', 'layers', 'add_children_to_output',
            'add_parents_to_output', 'return', 'regs', 'training', 'kernel',
            'filter', 'drop', 'act', 'bn', 'single_output', 'output', 'n', 'inference',
            'flatten_name', 'func', 'name', 'additive_drop', 'mode', 'weights',
           } if keys is None else keys
    layer_keys = layer.keys()
    for key in keys:
        if key not in layer_keys:
            layer[key] = None
    return layer


def list_select_models(models_dir=None):
    '''lists trained models and returns path of the chosen one.
    Args:
        models_dir: a directory path (str or pathlib.Path instance) where models were saved.
    Returns:
        (str) absolute path of the selected model.
    '''
    
    to_int = lambda list_x: [int(x) for x in list_x]
    
    if models_dir is None:
        models_dir = consts.Paths.trained_models_dir
    
    model_paths = sorted(glob(str(models_dir / "*")))

    print("="*150)
    for i, path in enumerate(model_paths):
        print(f'[{i+1}]   {path}')
    print("="*150)
    
    while(True):
        value = input("Which?[number | comma seperated numbers [i-j] | number*freq [i*k]:")
        if len(value):
            break

    values = re.findall(r"[0-9]+-[0-9]+|[0-9]+[\s]*[*][\s]*[0-9]+|[0-9]+", value)
    indices = []
    for v in values:
        if '-' in v:
            low, high = to_int(v.split('-'))
            indices.extend(list(range(low, high + 1)))
            
        elif '*' in v:
            index, freq = v.split('*')
            indices = [int(index) for _ in range(int(freq))]

        else:
            indices.append(int(v))

    if len(indices) == 1:
        # Return the number itself instead of a single member list
        return model_paths[indices[0] - 1]
            
    return [model_paths[idx - 1] for idx in indices]


def check_create_path(path) -> None:
    '''Assess whether path does exists. Otherwise, creates it.
    Args:
        path: a string or pathlib.Path instance
    Returns: 
        None
    '''
    
    Path(path).mkdir(parents=True, exist_ok=True) if not Path(path).is_dir() else None


def find_layer_conf(layers_conf, layer_name: str) -> int:
    '''Finds configuration of a TensorFlow/Keras layer specified by its name.
    Args:
        layers_conf: list of configuration of all layers in a model
        layer_name: name of the desired layer
    Returns:
        index of the layer
    '''
    for i, layer_conf in enumerate(layers_conf):
        if layer_conf['name'] == layer_name:
            return i
    
    raise ValueError(f"No such layer '{layer_name}'.")

def find_downstream_layers(layers_conf, layer_name: str) -> list:
    '''Finds all downstream layers of a layer specified by its name.
    Args:
        layers_conf: list of configuration of all layers in a model
        layer_name: name of the desired layer
    Returns:
        list of downstream layers, where for each layer there is a tuple containing its in_nodes and its config.
    '''
    downstream_layers = []
    for idx_in_conf, layer_conf in enumerate(layers_conf):
        if len(layer_conf['inbound_nodes']):
            inbound_nodes = layer_conf['inbound_nodes'][0]
            for idx_in_nodes, node in enumerate(inbound_nodes):
                if node[0] == layer_name:
                    downstream_layers.append([idx_in_nodes, idx_in_conf])
        
    return downstream_layers

def remove_layers(model, layers_name: list, add_children_to_output=False, add_parents_to_output=False, custom_objects=None, return_mode:str='model'):
    ''' Removes specific layers from a tf.keras model.
    Arguments:
        model: tf.keras model
        layers_name: (list) a list of strings (str) representing each layer's name to be removed.
        add_children_to_output: (bool) whether add downstream (children) layers to output of the model or not.
        custom_objects: (dict) if the model has custom layers, submodels, etc., shall be passed through this argument.
        return_mode: (str) can be 'model' or 'config'. if config, it returns updated model config, otherwise it return updated model.
    Returns:
        depending on the parameter 'return_mode', can be config of the updated model or the updated model itself
    '''
    
    assert return_mode in ['model', 'config'], "The parameter 'return_mode' must be 'model' or 'config'."
    
    config = model.get_config()
    
    def remove_single_layer(idx):
        '''Removes a single layer specified by its index in a model's layers configuration.''' 
        layer_conf = config['layers'][idx]
        layer_name = layer_conf['name']

        downstream_layers_confs = find_downstream_layers(config['layers'], layer_name)

        for idx_in_nodes, idx_in_conf in downstream_layers_confs:
            config['layers'][idx_in_conf]['inbound_nodes'][0].pop(idx_in_nodes)
            if len(layer_conf['inbound_nodes']):
                for inbound_node in layer_conf['inbound_nodes'][0]:
                    config['layers'][idx_in_conf]['inbound_nodes'][0].append(inbound_node)

        # Remove from outputs
        for i, output in enumerate(config['output_layers']):
            if output[0] == layer_name:
                config['output_layers'].pop(i)
                break
        # Add children to output
        if add_children_to_output:
            for idx_in_nodes, idx_in_conf in downstream_layers_confs:
                name = config['layers'][idx_in_conf]['name']
                config['output_layers'].append([name, 0, 0])

        # Add parents to output
        if add_parents_to_output:
            for node in layer_conf['inbound_nodes'][0]:
                config['output_layers'].append(node[:-1])

        # Remove layer
        config['layers'].pop(idx)
    
    idxs = []
    for layer_name in layers_name:
        idxs.append(find_layer_conf(config['layers'], layer_name))
    idxs = sorted(idxs)[::-1]
    for idx in idxs:
        remove_single_layer(idx)
    
    if return_mode == 'config':
        return config
    
    # Load weights
    new_model = tf.keras.Model().from_config(config, custom_objects)
    for layer in new_model.layers:
        layer.set_weights(model.get_layer(layer.name).get_weights())
        
    return new_model


def add_new_layer(new_layer, 
                  model,
                  upstream_layer_name: str,
                  call_method_kwargs: dict={},
                  inbound_nodes_args: list=None, 
                  custom_objects: dict=None,
                  add_to_output: bool=False,
                  load_weights: bool=True):
    

    config = model.get_config()

    new_layer_conf = new_layer.get_config()
    
    new_layer_conf = {
        'class_name': type(new_layer).__name__,
        'config': new_layer.get_config(),
        'name': new_layer.name,
        'inbound_nodes': [[]]}
    
    if inbound_nodes_args is None:
        inbound_nodes = [new_layer.name, 0, 0, {}]
        
    elif (len(inbound_nodes_args) != 3 or 
          not isinstance(inbound_nodes_args[0], int) or
          not isinstance(inbound_nodes_args[1], int) or
          not isinstance(inbound_nodes_args[2], dict)):
        
        raise ValueError(f"inbound_nodes_args must be a list of [int, int, dict] types got {inbound_nodes_args}.")
        
    else:
        inbound_nodes = inbound_nodes_args.copy()
        inbound_nodes.insert(0, new_layer.name)
        

    downstream_layers_confs = find_downstream_layers(config['layers'], upstream_layer_name)

    for idx_in_nodes, idx_in_conf in downstream_layers_confs:
        
        in_node = config['layers'][idx_in_conf]['inbound_nodes'][0].pop(idx_in_nodes)
        in_node[-1].update(call_method_kwargs)
        
        try:
            new_layer_conf['inbound_nodes'][0].index(in_node)
            
        except ValueError as ve:
            if 'not in list' in ve.args[0].lower():     
                new_layer_conf['inbound_nodes'][0].append(in_node)
                
            else:
                raise(ve)
        
#         new_layer_conf['inbound_nodes'][0][-1][-1].update(call_method_kwargs)
        
        config['layers'][idx_in_conf]['inbound_nodes'][0].append(inbound_nodes)
        
    # add new layer to config['layers'] list
    upstream_layer_idx = find_layer_conf(config['layers'], upstream_layer_name)
    config['layers'].insert(upstream_layer_idx + 1, new_layer_conf)
        
    # Add to network output
    if add_to_output:
        config['output_layers'].append(inbound_nodes[:-1])
        
    
    config['name'] = ''
    
#     return config
    
    # Create new model
    new_model = tf.keras.Model(name=None).from_config(config, custom_objects)

    # Load weights
    if not load_weights:
        return new_model
    
#     tf.keras.backend.clear_session()
    
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(layer.name).get_weights())
        except ValueError as ve:
            if 'no such layer' in ve.args[0].lower():
                continue
            else:
                raise(ve)

    return new_model

    
def add_prefix(*args, **kwargs):
    '''for compatibilty reasons'''
    return add_update_prefix(*args, **kwargs)
    

def add_update_prefix(model, prefix: str, old_prefix:str=None, custom_objects=None):
    '''Adds a prefix to layers and model name while 
    keeping the pre-trained weights.
    Args:
        model: a tf.keras model
        prefix: string representing prefix to be added to each layer
        old_prefix: if None, adds prefix. Otherwise, removes old_prefix and updates it with prefix parameter.
        custom_objects: (dict) if the model has custom layers, submodels, etc., shall be passed through this argument.
    '''
    config = model.get_config()
    old_to_new = {}
    new_to_old = {}

    for layer in config['layers']:
        if old_prefix is None:
            new_name = prefix + layer['name']
        else:
            remainder = "".join(re.split(old_prefix, layer['name']))
            new_name = prefix + remainder
            
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]

    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]

    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]

    
    if old_prefix is None:
        config['name'] = prefix + config['name']
    else:
        remainder = "".join(re.split(old_prefix, config['name']))
        config['name'] = prefix + "_" + remainder

    new_model = tf.keras.Model().from_config(config, custom_objects)
    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())

    return new_model


def data_generator(ds, set_: str, base_class: int):

    probability_array = np.ones(10) * .5/9
    probability_array[base_class] = 0.5
    set_ = set_.decode("utf-8")

    if set_ == 'train':
        interval = [0, 4900]
    elif set_ == 'val':
        interval = [4900, 5000]
    elif set_ == 'test':
        interval = [0, 1000]

    while True:
    # for i in range(16384):
        Y, img_ids = np.random.choice(10, p=probability_array), np.random.randint(interval[0], interval[1])
        X = ds[Y, img_ids]
        # yield X, keras.utils.to_categorical(Y, 10, dtype='uint8')
        yield X, int(Y == base_class)


def get_tf_dataset(dataset, base_class, set_):
    return tf.data.Dataset.from_generator(
        data_generator, 
        args=[dataset, set_, base_class],
        output_signature=(tf.TensorSpec(shape=(96, 96, 3), dtype=tf.uint8),
                          tf.TensorSpec(shape=(), dtype=tf.uint8)))


def get_optimized_dataset(tf_dataset, batch_size=128, do_augment=False):
    return (tf_dataset
            # .repeat()
            .map(lambda x, y: (preprocess(x, do_augment=do_augment), y),tf.data.AUTOTUNE)
            .batch(batch_size, drop_remainder=True)
            # .cache()
            .prefetch(tf.data.AUTOTUNE))


def augment(img, transform_idx):
    return [
        tfa.image.translate(img, tf.random.uniform([2], -20, 20), fill_mode='nearest'),
        tfa.image.sharpness(img, np.random.randint(-2, 2)),
        tf.image.random_brightness(img, .2),
        tf.image.random_contrast(img, .9, 1.1),
        tf.image.random_hue(img, .2),
        tf.image.random_saturation(img, 0.2, 1.2),
    ][transform_idx]


@tf.function
def preprocess(image, rescale=1/255, do_augment=True):
    """Preprocess function applied in dataset.map function"""
    image = tf.cast(image, tf.float32)
    if rescale:
        image *= rescale

    if do_augment:
        image = tf.image.random_flip_left_right(image)
        # [image, ] = tf.py_function(augment_fn, [image], [tf.float32])
        # transforms_inds = np.random.randint(0, 6, 2)
        # image = augment(img=image, transform_idx=transforms_inds[0])
        # image = augment(img=image, transform_idx=transforms_inds[1])
        image = augment(img=image, transform_idx=np.random.randint(6))
        image = tfa.image.rotate(image, tf.random.uniform([1], -.6, .6), fill_mode='nearest')
        image = tf.image.random_crop(tf.image.resize(image, (120, 120)), (96, 96, 3))
        # image = tf.image.random_crop(tf.image.resize(image, (110, 110)), (batch_size, 96, 96, 3))
        # image = tf.image.resize(tf.image.random_crop(image, (85, 85, 3)), (96, 96))
        image = tf.clip_by_value(image, -1, 1)
        # image = tf.clip_by_value(image, 0, 1)

    return image


def add_regularization(model, regularizer=tf.keras.regularizers.l2(1e-4)):
    ''' Adapted from https://sthalles.github.io/keras-regularizer/ '''
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
#         for attr in ['kernel_regularizer', 'activity_regularizer']:
#             if hasattr(layer, attr):
#                 setattr(layer, attr, regularizer)
        if isinstance(layer, tf.keras.layers.Conv2D):
            setattr(layer, 'kernel_regularizer', tf.keras.regularizers.l1_l2(l1=1e-8, l2=1e-8))
            setattr(layer, 'activity_regularizer', tf.keras.regularizers.l2(1e-8))
            setattr(layer, 'bias_regularizer', tf.keras.regularizers.l2(1e-8))

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model

# def insert_layers_in_model(model, layer_common_name, new_layer):
#     import re

#     layers = [l for l in model.layers]
# #     x = layers[0].output
#     layer_config = new_layer.get_config()
#     base_name = layer_config['name']
#     layer_class = type(new_layer)
#     for i in range(0, len(layers)):
# #         x = layers[i](x)
#         x = layers[i].output
#         match = re.match(".+" + layer_common_name + "+", layers[i].name)
#         # add layer afterward
#         if match:
#             layer_config['name'] = base_name + "_" + str(i)  # no duplicate names, could be done different
#             layer_copy = layer_class.from_config(layer_config)
#             x = layer_copy(x)

#     new_model = tf.keras.Model(inputs=layers[0].input, outputs=x)
#     return new_model


def insert_layer_nonseq(model, layer_regex, new_layer,
                        insert_layer_name=None, position='after'):
    ''' Partially adapted from https://stackoverflow.com/a/54517478'''
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
    new_layer_name_idx = 1
    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for i, layer in enumerate(model.layers[1:]):

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
#         if re.match(layer_regex, layer.name):
        if layer_regex in layer.name.lower():
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')
            layer_config = new_layer.get_config()
            base_name = layer_config['name']
            layer_class = type(new_layer)

            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:

                drop_rate = (.3 - (35 - new_layer_name_idx) * .01) if new_layer_name_idx > 15 else (new_layer_name_idx * .001)
                layer_config['rate'] = drop_rate
                new_layer_name_idx += 1

                layer_config['name'] = f'{base_name}_{drop_rate}_{new_layer_name_idx}'

                layer_copy = layer_class.from_config(layer_config)
            x = layer_copy(x)

            print(f'New layer: {x.name} backlink layer: {layer.name}')

            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    new_model = tf.keras.Model(inputs=model.inputs, outputs=model_outputs)

    # Fix possible problems with new model
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'temp_model_weights.h5')
    new_model.save(tmp_weights_path)
    new_model = tf.keras.models.load_model(tmp_weights_path)

    return new_model


def printd(msg: str, decorator='=', ncol=120):
    '''Decorates msg''' 
    print('\n', decorator*ncol, '\n', msg, '\n', decorator*ncol)
