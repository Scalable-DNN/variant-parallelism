import os, gc
import logging
from glob import glob
from datetime import datetime
from pathlib import Path
import shutil
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import yaml

import consts, utils

code_path = Path(os.path.realpath(__file__))
dir_path = code_path.parent
config_name = 'train_config.yml'
with open(dir_path / config_name, 'r') as yaml_file:
    tconf = yaml.load(yaml_file, Loader=yaml.FullLoader)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(tconf['gpu_id'])
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf

# Avoid fully occupation of gpu memory
if tconf['gpu_id'] > -1:
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)


import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers as KL
from tensorflow.keras import Model
import tensorflow_addons as tfa
from tensorflow.keras import applications as KA
import larq as lq
import tensorflow_datasets as tfds

from dataset import Dataset


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
conv_kernel = [3, 4, 5, 6, 7, 10]


@tf.function
def get_loss(y_true, y_pred, target_class, loss_type='binary'):

    def multi_class_loss():
        target_true = y_true[..., (target_class*5):(target_class+1)*5]
#         binary_target_true = tf.argmax(target_true, axis=-1)
#         return tf.keras.losses.binary_crossentropy(binary_target_true, y_pred)
        return tf.keras.losses.categorical_crossentropy(target_true, y_pred)

    def binary_loss():
        target_true = y_true[..., (target_class):(target_class+1)]
        return tf.keras.losses.binary_crossentropy(target_true, y_pred)

    return binary_loss() if loss_type == 'binary' else multi_class_loss()


def get_losses(coeffs, alpha: list):
    losses = {}
    loss_weights = {}
    sum_w_losses = 0
    for i in range(len(input_shapes)):
        pred_head_name = tconf['model_prefix'] % i + 'single_pred'
        losses[pred_head_name] = tf.keras.losses.CategoricalCrossentropy()
        loss_weights[pred_head_name] = coeffs[0]
#         w_loss = tconf['input_shapes'][i] / max(tconf['input_shapes']) * (alpha[i] / .35)
#         sum_w_losses += w_loss
#         loss_weights[pred_head_name] = w_loss
        

    losses['final_prediction'] = tf.keras.losses.CategoricalCrossentropy()
    loss_weights['final_prediction'] = coeffs[1]
#     loss_weights['final_prediction'] = sum_w_losses * 2
    
    utils.printd(f'Loss Weights: {loss_weights}')

    return losses, loss_weights


def bn_act_drop(x, drop=None, act='relu', bn=True, name_prefix=''):
    name = layer_name_handler(name_prefix, 'bn', i)
    x = KL.BatchNormalization(name=name)(x) if bn else x
    
    name = layer_name_handler(name_prefix, 'act', i)
    try:
        x = KL.Activation(act, name=name)(x) if act else x
    except ValueError:
        acts = {
            'relu6': tf.nn.relu6,
            'swish': tf.nn.swish,
        }
        x = KL.Activation(acts[act], name=name)(x) if act else x
        
    name = layer_name_handler(name_prefix, 'drop', i) + f'_{drop:.2f}'
    x = KL.Dropout(drop, name=name)(x) if drop else x
    return x


def reg(type_, value):
    value = [value] if not isinstance(value, list) else value
    return {
        'l1_l2': tf.keras.regularizers.l1_l2,
        'l2': tf.keras.regularizers.l2,
        'l1': tf.keras.regularizers.l1
    }[type_](*value)


def conv(x, type_, filter_, kernel, drop=None, act='relu', bn=None,
         regs=[1e-5]*4, name_prefix=''):
    name = layer_name_handler(name_prefix, type_, i)
    layer = {'sepconv': KL.SeparableConv2D,
             'conv': KL.Conv2D}[type_]
    if regs != -1:
        x = layer(filter_, kernel,
                  kernel_regularizer=reg('l1_l2', regs[:2]),
                  bias_regularizer=reg('l2', regs[2]),
                  activity_regularizer=reg('l2', regs[3]),
                  name=name)(x)
    else:
        x = layer(filter_, kernel, name=name)(x)
    x = bn_act_drop(x, drop, act, bn, name_prefix)
    return x



norm_layer = KL.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 3)
var = mean ** 2
augment = tf.keras.Sequential(
    [KL.experimental.preprocessing.RandomFlip("horizontal"),
     KL.experimental.preprocessing.RandomRotation(0.05),
#      KL.experimental.preprocessing.RandomRotation(0.01),
#      KL.experimental.preprocessing.RandomZoom((-.001, .001), (-.001, .001), 'nearest')
    ], name='augment')


mergers = {'add': KL.Add, 'avg': KL.Average,
           'concat': KL.Concatenate, 'max': KL.Maximum,
          }


def keys_validator(layer: dict, keys: set=None):
    keys = {'type', 'interpolation', 'new_size', 'alpha', 'include_top',
            'input_shapes', 'trainable', 'layers', 'add_children_to_output',
            'add_parents_to_output', 'return', 'regs', 'training', 'kernel',
            'filter', 'drop', 'act', 'bn', 'single_output', 'output', 'n',
            'flatten_name', 'func', 'name', 'additive_drop', 'mode', 'weights',
           } if keys is None else keys
    layer_keys = layer.keys()
    for key in keys:
        if key not in layer_keys:
            layer[key] = None
    return layer


# def pick_core_net(net, *args, **kwargs):
#     nets = {
#         'mobilenetV2': none,
#         'efficientnetB0':
#     }

class SingleOutputDrop(KL.Layer):
    def __init__(self, input_dim=200, n_models=2, max_batch_size=512, mode='drop', name=None):
        super(SingleOutputDrop, self).__init__(name=name)
        self.input_dim = input_dim
        self.n = n_models
        self.single_output_len = input_dim // n_models
#         self.zero_tensor = tf.zeros((max_batch_size, self.single_output_len))

    def call(self, inputs):
        
        output = []
        idx_drop, idx_fill = np.random.randint(self.n, size=2)
    
        for i in range(self.n):
            
            if i == idx_drop:
#                 output.append(self.zero_tensor[:inputs.shape[0], ...])
                output.append(inputs[..., idx_fill*self.single_output_len:(idx_fill+1)*self.single_output_len])
            else:
                output.append(inputs[:, i*self.single_output_len:(i+1)*self.single_output_len])
        
        output = tf.reshape(tf.stack(output, 1), (-1, self.input_dim))
#         tf.print(output.shape)
        
        return output
    
    
class SingleOutputScaler(KL.Layer):
    def __init__(self, factor, name=None):
        super(SingleOutputScaler, self).__init__(name=name)
        self.factor = factor
        
    def call(self,tensor):
        return tensor * self.factor
    

class Scaler(KL.Layer):
    
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Scaler, self).__init__(**kwargs)   
        self.output_dim = output_dim
        self.input_dim = input_dim
        
    def build(self, input_shapes):
        super(Scaler, self).build(input_shapes)
#         self.DotLayer = KL.Dot(axes=(1,2))
        self.kernel = self.add_weight(name='kernel', shape=(self.input_dim, self.output_dim), initializer='uniform', trainable=True)
#         tile = tf.constant([input_shapes[0], 1, 1])
#         self.kernel_tiled = tf.tile()
        
    def call(self, x):
        return tf.keras.backend.expand_dims(self.kernel, 0)
#         return self.kernel

    def compute_output_shape(self):
        return self.output_dim
    
    
class Decompressor(KL.Layer):
    def __init__(self, output_dim, **kwargs):
        super(Decompressor, self).__init__(**kwargs)
        self.output_dim = output_dim
        
    def build(self, output_dim):
#         super(Decompressor, self).build(inputs)
#         self.wc = self.add_weights(name="compressor_weights", shape=(input_shape[0], input_shape[1], n_output))
        self.output_dim = output_dim
#         self.input_shape
#         self.row_tensor = tf.tile(tf.range(tf.shape(x)[0])[:, tf.newaxis], (1, layer['n']))
    
    def call(self, compressed, compressed_indices):
        
        row_tensor = tf.tile(tf.range(tf.shape(compressed)[0])[:, tf.newaxis], (1, tf.shape(compressed)[-1]))
        compressed_indices = tf.stack([row_tensor, compressed_indices], axis=-1)
        return tf.scatter_nd(compressed_indices, compressed, 100)
        
    

layer_counter = {'dense': 0, 'sepconv': 0, 'aug': 0, 'merger': 0,
                 'conv': 0, 'globalavg': 0, 'reshape': 0,
                 'act': 0, 'bn': 0, 'drop': 0, 'globalmax': 0,
                 'resize': 0, 'randcrop': 0, 'flatten': 0, 
                }
current_model = 0

def layer_name_handler(prefix, name, model_idx):
    global current_model
    if model_idx != current_model:
        for key in layer_counter.keys():
            layer_counter[key] = 0
        current_model = model_idx
    layer_counter[name] += 1
    return prefix + name + str(layer_counter[name])
    

def layer_case_decider(case, x=None, name_prefix=None):
    case = str(case.lower())

    if case == 'dense':
        name = layer['name']
        if layer['output'] and name is None:
            name = tconf['model_prefix'] % i + 'single_pred'
        else:
            name = layer_name_handler(name_prefix, case, i)
        x = KL.Dense(layer['n'], name=name, activation=layer['act'])(x)
        if layer['drop']:
            name = layer_name_handler(name_prefix, 'drop', i)
            x = KL.Dropout(layer['drop'], name=name)(x)      
    elif case == 'reshape':
        name = layer_name_handler(name_prefix, case, i)
        x = KL.Reshape([int(v) for v in layer['shape'].split()], name=name)(x)
    elif case == 'base_model':
        x = x[1](x[0], training=layer['training'])
#     elif case == 'core_net':
    elif case == 'mobilenetv2':
        input_shape = layer['input_shapes'][i]
        alpha = layer['alpha'][i]
        depthwise_multipliers.append(alpha)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(input_shape, input_shape, 3),
            alpha=alpha, include_top=layer['include_top'])
        base_model.trainable = layer['trainable']
        single_model_names.append(base_model.name)
        x = [x, base_model]
    elif case == 'remove':
        x[1] = utils.remove_layers(
            x[1],
            layer['layers'],
            add_children_to_output=layer['add_children_to_output'],
            add_parents_to_output=layer['add_parents_to_output'],
            return_mode=layer['return'])
    elif case == 'add_reg':  # add regularizations to conv layers of core network
        x[1] = utils.add_regularization(x[1])
    elif case == 'add_prefix':
        prefix = tconf['model_prefix'] % i
        x[1] = utils.add_prefix(x[1], prefix)
        single_model_names[i] = prefix + single_model_names[i]
    elif case in ['resizing', 'resize']:
        intrp = layer['interpolation'][i]
        new_shape = int(layer['new_size'][i])
        name = layer_name_handler(name_prefix, 'resize', i)
        x = KL.experimental.preprocessing.Resizing(new_shape, new_shape, intrp, name=name)(x)
    elif case == 'random_crop':
        new_shape = int(layer['size'][i])
        name = layer_name_handler(name_prefix, 'randcrop', i)
        x = KL.experimental.preprocessing.RandomCrop(new_shape, new_shape, name=name)(x)
    elif case == 'globalavg':
        name = layer_name_handler(name_prefix, case, i)
        x = KL.GlobalAveragePooling2D(name=name)(x)
    elif case == 'globalmax':
        name = layer_name_handler(name_prefix, case, i)
        x = KL.GlobalMaxPooling2D(name=name)(x)
    elif case in ['sepconv', 'conv']:
        drop = layer['drop'] if layer['drop'] else 0
        regs = layer['regs'] if layer['regs'] else -1
        filter_ = layer['filter']
        if isinstance(filter_, list):
            filter_ = filter_[i]
        if isinstance(layer['kernel'], int):
            k = layer['kernel']
        else:
            k = tuple(int(v) for v in layer['kernel'].split())
            
        if layer['additive_drop']:
            drop += layer['additive_drop'][i]
            
        k = x.shape[-2] if k == 0 else k

        if isinstance(regs, str):
            regs = [float(reg) for reg in regs.split()]
        act = layer['act'] if layer['act'] else None
        bn = layer['bn'] if layer['bn'] else None
        x = conv(x, case, filter_, k, drop, act=act, bn=bn, regs=regs, name_prefix=name_prefix)
        if layer['output']:
            name = layer['flatten_name'] if layer['flatten_name'] \
                else tconf['model_prefix'] % i + 'single_pred'
            x = KL.Flatten(name=name)(x)
    elif case == 'merger':
        merge_layer = mergers[layer['func'].lower()]
        name = layer['name']
        if name is None:
            name = layer_name_handler(name_prefix, 'merger', i)
        x = merge_layer(name=name)(x)
    elif case == 'act':
        x = KL.Activation(layer['act'], name=layer['name'])(x)
    elif case == 'model_drop':
        x = KL.Lambda(model_drop, name=f'ModelDrop_{layer["drop"]}',
                      arguments={'drop': layer['drop'],
                                 'n_models': tconf['network']['n']})(x)
    elif case == 'single_output_drop':
        x = SingleOutputDrop(input_dim=tconf['network']['ensemble'][-1]['n'], 
                             n_models=tconf['network']['n'],
                             mode=layer['mode'],
                             name='SingleOutputDrop')(x)
    elif case == 'output_scaler':
        factor = tconf['input_shapes'][i] / max(tconf['input_shapes']) * (depthwise_multipliers[i] / .35) ** 2
        name = name_prefix + 'output_scaler'
        x = SingleOutputScaler(factor, name=name)(x)
    elif case == 'output_compressor':
#         n_original = x.shape[-1]
#         name = layer_name_handler(name_prefix, 'flatten', i)
#         x = KL.Flatten(name=name)(x)
#         output_comp = KL.Dense(layer['n'], name=name_prefix+'compress')(x)
#         scaler = KL.Dense(layer['n'], name=name_prefix+'scaler')(x)
#         multiplied = KL.Multiply(name=name_prefix+'multiply')([output_comp, scaler])
#         multiplied = KL.Activation(tf.nn.relu6, name=name_prefix+'multiply_relu')(multiplied)
#         name = name_prefix+'decompressor'
#         if layer['output']:
#             name = tconf['model_prefix'] % i + 'single_pred'
#         output_decomp = KL.Dense(n_original, name=name, activation='softmax')(multiplied)
#         x = output_decomp
        
        compressed, compressed_indices = tf.nn.top_k(x, k=layer['n'], sorted=False, name=name_prefix+f'Top_{layer["n"]}')
        compressed_softmax = tf.nn.softmax(compressed, name=name_prefix+"Softmax")
        shape_x = tf.shape(x)
        row_tensor = tf.tile(tf.range(shape_x[0])[:, tf.newaxis], (1, layer['n']))
        indices = tf.stack([row_tensor, compressed_indices], axis=-1, name=name_prefix+f'Top_{layer["n"]}_Indices')
        x = tf.scatter_nd(indices, compressed_softmax, shape_x, name=name_prefix+"Decompressed")
        
        
    
        
    if layer['single_output']:
        single_outputs.append(x)
    if layer['output']:
        outputs.append(x)

    return x


single_outputs = []
outputs = []
single_model_names = []
depthwise_multipliers = []

input_size = [int(s) for s in tconf['network_input_size'].split(',')]
inputs = KL.Input(input_size, name='model_input')
normed = norm_layer(inputs)
norm_layer.set_weights([mean, var])
normed = augment(normed)
for i in range(tconf['network']['n']):
    x = normed
    layer_name_prefix = tconf['model_prefix'] % i + 'x_'
    for layer in tconf['network']['core']:
        layer = keys_validator(layer)
        x = layer_case_decider(layer['type'], x, layer_name_prefix)

x = single_outputs

for j, layer in enumerate(tconf['network']['ensemble']):
    layer = keys_validator(layer)
    if layer['type'] in ['merger', 'model_drop']:
        break
    for i in range(len(x)):
        layer_name_prefix = tconf['model_prefix'] % i + 'x_'
        x[i] = layer_case_decider(layer['type'], x[i], layer_name_prefix)

for layer in tconf['network']['ensemble'][j:]:
    layer_name_prefix = "EnsembleHead_"
    layer = keys_validator(layer)
    x = layer_case_decider(layer['type'], x, layer_name_prefix)


ensemble = Model(inputs, outputs, name='EnsembleModel')
ensemble.summary(160, [.45, .58, .65, 1])

lq.models.summary(ensemble)


def set_trainable_param(idx):
    for i, model_name in enumerate(single_model_names):
        base_model = ensemble.get_layer(model_name)
        base_model.trainable = True
#         utils.printd(base_model.name)
        for i, layer in enumerate(base_model.layers):
            if i < idx:
                layer.trainable = False
#             print(f"{i+1:<5}{layer.name:<50}",
#                   ['frozen', 'trainable'][int(layer.trainable)])

rand_aug = None
if 'rand_aug' in tconf['dataset'].keys():
    rand_aug = tconf['dataset']['rand_aug']

dataset = Dataset(tconf['dataset']['name'], rand_aug=rand_aug)
 

def train(callbacks, loss_coeffs, batch_size, epochs, lr, trainable_idx):

    losses, loss_weights = get_losses(loss_coeffs, depthwise_multipliers)
    optimizer = tf.keras.optimizers.Adam(lr)
#     tfd_train, tfd_test = get_tfds(batch_size)
    if tconf['dataset']['type'] == 'mixup':
        tfd_train = dataset.mix_up(batch_size, alpha=tconf['dataset']['alpha'])
    else:
        tfd_train = dataset.get_tfds_train(batch_size)
    
#     tfd_train = tfd_train.map(lambda x, y: (rand_aug.distort(x), y), 
#                               num_parallel_calls=tf.data.AUTOTUNE)
        
    tfd_test = dataset.get_tfds_test(batch_size * 2)
    if trainable_idx is not None:
        set_trainable_param(trainable_idx)
    ensemble.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])
    ensemble.fit(tfd_train, validation_data=tfd_test, epochs=epochs, callbacks=callbacks, verbose=2)


def get_callbacks(cb_keys, monitor='val_final_prediction_accuracy'):
    logs = str(model_path / 'logs')
    cbs = {
        'RoLRP': tf.keras.callbacks.ReduceLROnPlateau('final_prediction_accuracy', factor=.98, patience=0, verbose=1),
        'ES': tf.keras.callbacks.EarlyStopping('final_prediction_accuracy', patience=20),
        'MC': tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor, save_best_only=True),
        'TB': tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, 
#                                              profile_batch=(50, 70),
                                            )
    }
    
    return [cbs[key] for key in cb_keys]


## Save configs, code, and model plot
# ================================================================================
utils.check_create_path(model_path)
tf.keras.utils.plot_model(ensemble, to_file=str(model_path / 'plot.png'),
                          show_shapes=True, expand_nested=True)
shutil.copy(str(code_path), str(model_path / 'training_code.py'))
shutil.copy(str(dir_path / config_name), str(model_path / config_name))
# ================================================================================


def evaluate(do_save=False):
    tfd_test = dataset.get_tfds_test(tstep['batch'] * 2, max_n=-1)
    evs = ensemble.evaluate(tfd_test, verbose=0)
    msg = ''
    msg += f' {"Ensemble":<30}{evs[-1]*100:.2f}%' + '\n'
    for i, name in enumerate(single_model_names[::-1]):
        msg += f' {name:<30}{evs[-(i+2)]*100:.2f}%' + '\n'
        
    # Save
    if do_save:
        gc.collect()
        with open(model_path / 'evaluation.txt', 'w') as file:
            file.write(msg)
        gc.collect()
    
    msg = 'Evaluation Step:' + '\n' + msg[:-2]
    utils.printd(msg, '#')


# Training steps
start_time = datetime.now()
for tstep in tconf['train']:
    gc.collect()
    utils.printd(model_name, '*')
    utils.printd(tstep)
    callbacks = get_callbacks(tstep['callbacks'].split())
    loss_coeffs = [float(v) for v in tstep['loss_coeffs'].split()]
    train(callbacks, loss_coeffs, tstep['batch'], tstep['epoch'],
          tstep['lr'], tstep['trainable_idx'])

    elapsed = datetime.now()
    utils.printd('Elapsed Time: '+''.join(str((elapsed - start_time)).split('.')[:-1]))
    
    do_save = 'save' in tstep.keys() and tstep['save']
    
    # Evaluate
    gc.collect()
    evaluate(do_save)
    
    # Save model
    if do_save:
        print("Saving Model ...")
        ensemble.save(str(model_path))
