import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import Model
from larq import models as lq_models

import custom_layers as CL
import utils


class NetworkConstructor:
    
    def __init__(self, input_shape, single_input_shapes, model_prefix, name_prefix=''):
        self.model_prefix = model_prefix
        self.name_prefix = name_prefix
        self.layer_counter = {}
        self.current_model = 0
        self.current_model_idx = 0
        self.input_shape = input_shape
        self.single_input_shapes = single_input_shapes
        self.single_outputs = []
        self.outputs = []
        self.single_model_names = []
        self.depthwise_multipliers = []
        self.single_outputs_name = []
        self.final_outputs_name = []
        
        self.acts = {'relu': tf.nn.relu, 
                     'relu6': tf.nn.relu6, 
                     'swish': tf.nn.swish, 
                     'softmax': tf.nn.softmax}
        
        self.mergers = {'add': KL.Add, 
                        'avg': KL.Average, 
                        'concat': KL.Concatenate, 
                        'max': KL.Maximum}
        
        self.model_input = KL.Input(self.input_shape, name='model_input')
        self.x = self.model_input
        
    def set_curennt_model_idx(self, idx):
        self.current_model_idx = idx
        
    def get_outputs_names(self):
        return self.single_outputs_name, self.final_outputs_name
        
    def set_x(self, new_x):
        self.x = new_x
    
    def get_x(self):
        return self.x
    
    def set_x_single_output(self, idx=None):
        self.x = self.single_outputs[idx] if idx is not None else self.single_outputs
        
    def set_name_prefix(self, name_prefix):
        self.name_prefix = name_prefix
    
    def layer_name_handler(self, name: str) -> str:
        '''Handles naming policy of layers.
        Args:
            name: name of the layer
        Returns:
            generated name for the layer
        '''
        if self.current_model_idx != self.current_model:
            self.layer_counter = {name: 0}
            self.current_model = self.current_model_idx
            
        if name in self.layer_counter.keys():
            self.layer_counter[name] += 1
        else:
            self.layer_counter[name] = 1
            
        
        return self.name_prefix + name + str(self.layer_counter[name])
    
    def augmentation(self, augmentations, inference=False, name='augment'):
        '''Generates augmentation layers (with tf.keras.Sequential)
        Args:
            augmentations: a list of dictionaries which their key is name of the augmentation function and 
                           their values are parameters required.
            name: name of the layer. Default: augment
        '''
        
        if inference:
            utils.printd("Augmentation in Inference Mode.")
            self.x = CL.RandAugment()(self.x)
            return
        
        aug_list = []
        prep_layers = KL.experimental.preprocessing
        for aug in augmentations:
            func = list(aug.keys())[0].lower()
            if 'random_flip' in func:
                aug_list.append(prep_layers.RandomFlip(aug[func]['mode']))
            elif 'random_rotation' in func:
                aug_list.append(prep_layers.RandomRotation(aug[func]['prob']))
            elif 'random_zoom' in func:
                probs = [float(x) for x in aug[func]['prob'].split()]
                aug_list.append(prep_layers.RandomZoom(
                    probs[:2], probs[2:], aug[func]['interpolation']))
        
        self.x = tf.keras.Sequential(aug_list, name=name)(self.x, training=True)
    
    def bn_act_drop(self, drop=0, act='relu', bn=True):
        '''Optioanlly inserts the following layers into the model, in order:
        BatchNormalization, Activation, Dropout.
        Args:
            drop: a float having range [0,1] determining probabilty of random dropout. If None, no dropout layer will be insterted.
            act: a string detrmining activation function. current supported functions: [relu, relu6, swish].
                 If None, no activation layer will be inserted.
                 You can add more activation functions withing __init__ method by modifying the dictionary 'self.acts'.
            bn: a boolean determining whether to apply batch normalization or not.
        '''
        name = self.layer_name_handler('bn')
        self.x = KL.BatchNormalization(name=name)(self.x) if bn else self.x
        
        name = self.layer_name_handler('act')
        self.x = KL.Activation(self.acts[act], name=name)(self.x) if act else self.x
        
        name = self.layer_name_handler('drop') + f'_{drop:.2f}'
        self.x = KL.Dropout(drop, name=name)(self.x, training=True) if drop else self.x
        
    def regulizers(self, type_, value):
        '''Generates regularization terms for a layer
        Args:
            type_: a string determining regularization type. It can be one of ['l1_l2', 'l2', 'l1']
            value: a tuple of floats or a float having range of [0, 1], determining regularization(s) value.
        Returns: generated regularization term  
        '''
        value = [value] if not isinstance(value, list) else value
        return {
        'l1_l2': tf.keras.regularizers.l1_l2,
        'l2': tf.keras.regularizers.l2,
        'l1': tf.keras.regularizers.l1
        }[type_](*value)
    
    def conv(self, type_, filter_:int, kernel, regs=[1e-5]*4, **bn_act_drop_kwargs):
        '''Inserts a convolution layer into the model.
        Args:
            type_: can be one of ['sepconv', 'conv']
            filter_: number of output filters
            regs: a tuple of four floating points determining following regularization terms:
                1) l1_l2 kernel_regularizer (first two floats)
                2) l2 bias_regularizer (third float)
                3) l2 activity_regularizer (fourth float)
            **bn_act_drop_kwargs: arguments to be used in the 'bn_act_drop' method. 
                                  For more details, refer to the respective method.
        '''
        layer = {'sepconv': KL.SeparableConv2D, 'conv': KL.Conv2D}[type_]
        name = self.layer_name_handler(type_)
        if regs != -1:
            self.x = layer(filter_, kernel,
                           kernel_regularizer=reg('l1_l2', regs[:2]),
                           bias_regularizer=reg('l2', regs[2]),
                           activity_regularizer=reg('l2', regs[3]),
                           name=name)(self.x)
        else:
            self.x = layer(filter_, kernel, name=name)(self.x)
        
        self.bn_act_drop(**bn_act_drop_kwargs)
        
    def layer_case_decider(self, layer):
        '''Depending on the "case", decides what layer to generate. 
        Args:
            layer: a dictionary containing parameters required to generated a layer.
        Return:
            output of generated layer.
        '''
        case = str(layer['type'].lower())

        if case == 'dense':
            name = layer['name']
            if layer['output'] and name is None:
                name = self.model_prefix % self.current_model_idx + 'single_pred'
            else:
                name = self.layer_name_handler(case)
                
            self.x = KL.Dense(layer['n'], name=name, activation=layer['act'])(self.x)
            
            if layer['drop']:
                name = self.layer_name_handler('drop')
                self.x = KL.Dropout(layer['drop'], name=name)(self.x, training=True)   
                
        elif case == 'normalization':
            mean = [127.5] * 3 if layer['mean'] == 'auto' else layer['mean']
            mean = np.array(mean)
            var = mean ** 2 if layer['var'] == 'auto' else np.array(layer['var'])
            # self.norm_layer = KL.experimental.preprocessing.Normalization()
            self.norm_layer = KL.Normalization(mean=mean, variance=var)
            self.x = self.norm_layer(self.x)
            # self.norm_layer.set_weights([mean, var])
            
        elif case == 'rescale':
            name = self.layer_name_handler(case)
            self.x = KL.experimental.preprocessing.Rescaling(layer['scale'], layer['offset'], name=name)(self.x)
            
        elif case == 'augmentation':
            self.augmentation(layer['func'], inference=layer['inference'])
            
        elif case == 'reshape':
            name = self.layer_name_handler(case)
            self.x = KL.Reshape([int(v) for v in layer['shape'].split()], name=name)(self.x)
            
        elif case == 'base_model':
            self.x = self.x[1](self.x[0], training=layer['training'])
            
    #     elif case == 'core_net':
    
        elif case == 'multiple_models':
#             model_outputs = []
#             for model in layer['models']:
#                 model_outputs.append(model(self.x))
            self.x = [model(self.x) for model in layer['models']]
        
        elif case == 'meta_architecture':
            apps = tf.keras.applications
            meta_arch = {
                'mobilenetv2': apps.MobileNetV2,
                'mobilenetv3small': apps.MobileNetV3Small,
                'mobilenetv3large': apps.MobileNetV3Large,
                'resnet50': apps.ResNet50,
                'efficientnetb0': apps.EfficientNetB0,
                'nasnetmobile': apps.NASNetMobile,
            }
            
            input_shape = self.single_input_shapes[self.current_model_idx]
            
            if layer['alpha'] is None:
                layer['alpha'] = 1.0
            self.depthwise_multipliers = layer['alpha']
            if not isinstance(self.depthwise_multipliers, list):
                self.depthwise_multipliers = [self.depthwise_multipliers] * len(self.single_input_shapes)
                
            alpha = self.depthwise_multipliers[self.current_model_idx]
            
            model_args = {
                'input_shape': (input_shape, input_shape, 3),
                'include_top': layer['include_top'],
            }
            
            if 'mobilenet' in layer['arch']:
                model_args['alpha'] = alpha
            
            if 'mobilenetv3' in layer['arch'] and 'minimalistic' in layer.keys():
                model_args['minimalistic'] = layer['minimalistic']
            
            base_model = meta_arch[layer['arch']](**model_args)
            
            base_model.trainable = layer['trainable']
            
            self.single_model_names.append(base_model.name)
            self.x = [self.x, base_model]

        elif case == 'mobilenetv2':
            input_shape = self.single_input_shapes[self.current_model_idx]
            self.depthwise_multipliers = layer['alpha']
            if not isinstance(self.depthwise_multipliers, list):
                self.depthwise_multipliers = [self.depthwise_multipliers] * len(self.single_input_shapes)

            alpha = self.depthwise_multipliers[self.current_model_idx]

            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(input_shape, input_shape, 3),
                alpha=alpha, 
                include_top=layer['include_top'])

            base_model.trainable = layer['trainable']

            self.single_model_names.append(base_model.name)
            self.x = [self.x, base_model]

        elif case == 'insert_dropout':

            new_model = tf.keras.models.clone_model(self.x[1])
            new_model.set_weights(self.x[1].get_weights())

            insertion_indices = []
            for j, l in enumerate(self.x[1].get_config()['layers']):
                if layer['after_name_parttern'] in l['class_name']:
                    insertion_indices.append(j)

            for j, idx in enumerate(insertion_indices):
                new_layer = KL.Dropout(0.05)
                parent_name = self.x[1].layers[idx].name
                new_model = utils.add_new_layer(
                    new_layer=new_layer,
                    model=self.x[1],
                    upstream_layer_name=parent_name,
                    call_method_kwargs=layer['call_method_kwargs'],
                    inbound_nodes_args=[0, 0, {}],
                    custom_objects=None,
                    add_to_output=layer['output'],
                    load_weights=True)

            self.x[1] = new_model

        elif case == 'remove':
            self.x[1] = utils.remove_layers(
                self.x[1],
                layer['layers'],
                add_children_to_output=layer['add_children_to_output'],
                add_parents_to_output=layer['add_parents_to_output'],
                return_mode=layer['return'])
            
        elif case == 'add_reg':  # add regularizations to conv layers of core network
            self.x[1] = utils.add_regularization(self.x[1])
            
        elif case == 'add_prefix':
            prefix = self.model_prefix % self.current_model_idx
            self.x[1] = utils.add_update_prefix(self.x[1], prefix)
            self.single_model_names[-1] = prefix + self.single_model_names[-1]
            
        elif case in ['resizing', 'resize']:
            intrp = layer['interpolation'][self.current_model_idx]
            new_shape = int(layer['new_size'][self.current_model_idx])
            name = self.layer_name_handler('resize')
            
            self.x = KL.experimental.preprocessing.Resizing(new_shape, new_shape, intrp, name=name)(self.x)
            
        elif case == 'random_crop':
            new_shape = int(layer['size'][self.current_model_idx])
            name = self.layer_name_handler('randcrop')
            if layer['tflite']:
                self.x = CL.RandomCrop2D(new_shape, name=name)(self.x, training=True)
            else:
                self.x = KL.experimental.preprocessing.RandomCrop(new_shape, new_shape, name=name)(self.x, training=True)
            
        elif case == 'globalavg':
            name = self.layer_name_handler(case)
            self.x = KL.GlobalAveragePooling2D(name=name)(self.x)
            
        elif case == 'globalmax':
            name = self.layer_name_handler(case)
            self.x = KL.GlobalMaxPooling2D(name=name)(self.x)
            
        elif case in ['sepconv', 'conv']:
            drop = layer['drop'] if layer['drop'] else 0
            regs = layer['regs'] if layer['regs'] else -1
            
            filter_ = layer['filter']
            if isinstance(filter_, list):
                filter_ = filter_[self.current_model_idx]
                
            if isinstance(layer['kernel'], int):
                k = layer['kernel']
            else:
                k = tuple(int(v) for v in layer['kernel'].split())

            if layer['additive_drop']:
                drop += layer['additive_drop'][self.current_model_idx]

            k = self.x.shape[-2] if k == 0 else k

            if isinstance(regs, str):
                regs = [float(reg) for reg in regs.split()]
                
            act = layer['act'] if layer['act'] else None
            bn = layer['bn'] if layer['bn'] else None
            
            self.conv(case, filter_, k, regs=regs, drop=drop, act=act, bn=bn)
            
            if layer['output']:
                name = layer['flatten_name'] if layer['flatten_name'] \
                    else self.model_prefix % self.current_model_idx + 'single_pred'
                
                self.x = KL.Flatten(name=name)(self.x)
                
        elif case == 'merger':
            merge_layer = self.mergers[layer['func'].lower()]
            
            name = layer['name']
            if name is None:
                name = self.layer_name_handler('merger')
                
            self.x = merge_layer(name=name)(self.x)
            
        elif case == 'act':
            self.x = KL.Activation(layer['act'], name=layer['name'])(self.x)
            
        elif case == 'single_output_drop':
            # TODO: tconf is inaccessible here
            self.x = CL.SingleOutputDrop(
                input_dim=tconf['network']['ensemble'][-1]['n'],
                n_models=len(self.single_input_shapes),
                mode=layer['mode'],
                name='SingleOutputDrop')(self.x)

        elif case == 'output_scaler':
            if layer['factor'] == 'auto':
                factor = self.single_input_shapes[self.current_model_idx] / max(self.single_input_shapes)
#                 factor = self.single_input_shapes[self.current_model_idx] / 224
                factor *= (self.depthwise_multipliers[self.current_model_idx] / .35) ** 1.15
            elif isinstance(layer['factor'], list):
                factor = float(layer['factor'][self.current_model_idx])
            elif isinstance(layer['factor'], float):
                factor = layer['factor']
            else:
                raise "'factor' must be float or list of floats or 'auto'."
                
            name = self.name_prefix + 'output_scaler'
            utils.printd(f'{name}: {factor}')
            self.x = CL.SingleOutputScaler(factor, name=name)(self.x)
            
        elif case == 'output_compressor':
#             name_prefix = self.name_prefix
#             compressed, compressed_indices = tf.nn.top_k(self.x, k=layer['n'], sorted=False, name=name_prefix+f'Top_{layer["n"]}')
#             compressed_softmax = tf.nn.softmax(compressed, name=name_prefix+"Softmax")
#     #         x = Decompressor(100)(compressed, compressed_indices)
#             shape_x = tf.shape(self.x)
#             row_tensor = tf.tile(tf.range(shape_x[0])[:, tf.newaxis], (1, layer['n']))
#             indices = tf.stack([row_tensor, compressed_indices], axis=-1, name=name_prefix+f'Top_{layer["n"]}_Indices')
#             self.x = tf.scatter_nd(indices, compressed_softmax, shape_x, name=name_prefix+"Decompressed")
#             self.x = CL.DummyCompressor(layer['n'])(self.x)
            
            self.output_shape = tf.shape(self.x)
            self.k_in_top_k = layer['n']
            
            self.x = CL.Compression(
                k=layer['n'], 
                apply_softmax=layer['softmax'], 
                name=self.name_prefix + f"compressed_top{layer['n']}")(self.x)
            

#             self.x = compressed, compressed_indices #, tf.shape(self.x)
                
# #             compressed, compressed_indices = tf.nn.top_k(x, k=layer['n'], sorted=False, name=name_prefix+f'Top_{layer["n"]}')
# #             compressed_softmax = tf.nn.softmax(compressed, name=name_prefix+"Softmax")

#             shape_x = tf.shape(self.x)
#             row_tensor = tf.tile(tf.range(shape_x[0])[:, tf.newaxis], (1, layer['n']))
#             indices = tf.stack([row_tensor, compressed_indices], axis=-1, name=name_prefix+f'Top_{layer["n"]}_Indices')
#             x = tf.scatter_nd(indices, compressed_softmax, shape_x, name=name_prefix+"Decompressed")

        elif case == 'output_decompressor':
#             pass
#             self.x = CL.Decompression(shape=self.x[2], k=layer['n'], name=self.name_prefix + "decompressed")(self.x[0], self.x[1])
#             self.x = CL.Decompression(shape=self.output_shape, k=self.k_in_top_k, name=self.name_prefix + "decompressed")(self.x)
#             self.x = CL.decompression(self.x, self.output_shape, self.k_in_top_k, self.name_prefix + "decompressed")
    
            row_tensor = tf.tile(tf.range(self.output_shape[0], dtype=tf.int32)[:, tf.newaxis], (1, self.k_in_top_k))        
            indices = tf.stack([row_tensor, tf.cast(self.x[..., self.k_in_top_k:], tf.int32)], axis=-1)
            self.x = tf.scatter_nd(indices, self.x[..., :self.k_in_top_k], self.output_shape)
            
        elif case == 'old_output_compressor':
            compressed, compressed_indices = tf.nn.top_k(self.x, k=layer['n'], sorted=False)
            compressed_softmax = tf.nn.softmax(compressed)
            intermediate = tf.concat([compressed, tf.cast(compressed_indices, tf.float32)], axis=-1)
            shape_x = tf.shape(self.x)
            row_tensor = tf.tile(tf.range(shape_x[0])[:, tf.newaxis], (1, layer['n']))
            indices = tf.stack([row_tensor, intermediate[..., self.k_in_top_k:]], axis=-1)
            self.x = tf.scatter_nd(indices, intermediate[..., :self.k_in_top_k], shape_x)
            

        if layer['single_output']:
            self.single_outputs.append(self.x)
            self.single_outputs_name.append(self.x.name.split('/')[0])
        if layer['output']:
            self.outputs.append(self.x)
            self.final_outputs_name.append(self.x.name.split('/')[0])

        return self.x
    
    def create_model(self, model_name='EnsembleModel', show_summary=True):
        
        self.model = Model(self.model_input, self.outputs, name=model_name)
        
        if show_summary:
            self.model.summary(160, [.45, .58, .65, 1])
            lq_models.summary(self.model)
            
        return self.model
    
    def get_model(self):
        return self.model
    
    def set_trainable_param(self, model, idx: int):
        '''All layers before idx become frozen, and the rests will be trained.
        Args:
            model: a tf.keras model. If model is None, it uses self.model variable. Therefore, you have to call create_model method first.
            idx: index
        Return: a tf.keras model (It also saves the model in self.model)
        '''
                
        for i, model_name in enumerate(self.single_model_names):
            base_model = model.get_layer(model_name)
            base_model.trainable = True
            
            for i, layer in enumerate(base_model.layers):
                if i < idx:
                    layer.trainable = False
                # print(f"{i+1:<5}{layer.name:<50}", ['frozen', 'trainable'][int(layer.trainable)])
                
                # if i % 5 == 0:
                #     input()
        
        return model
