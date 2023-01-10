import tensorflow as tf
from tensorflow.keras import layers as KL

import utils


class RandAugment(KL.Layer):
    
    def __init__(self, **kwargs):
        super(RandAugment, self).__init__(**kwargs)
    
    def get_config(self):
        config = super(RandomAugment, self).get_config()
        
        return config
    
    def call(self, x):  
    
        shape_x = x.get_shape()
        value = tf.random.uniform([2], -0.2, 0.2)
        
        function = tf.stack([
            tfa.image.random_cutout(x, [8, 8]),
            tf.image.random_brightness(x, .2),
            tf.image.random_contrast(x, .02, .05),
            tf.image.random_flip_left_right(x),
            tf.image.random_saturation(x, .9, 1.1),
            tf.image.random_hue(x, .2),
            tfa.image.translate(x, [value[0] * shape_x[1] , value[1] * shape_x[2]]),
            tfa.image.rotate(x, value[1]),
            tfa.image.sharpness(x, 0.1),
            tfa.image.sharpness(x, 1.1),
        ])
        
        idx = tf.random.uniform([2], 0, function.shape[0], dtype=tf.int32)
        y = function[idx[0]]
        y = function[idx[1]]
        
        y.set_shape(shape_x)
        
        return y
    

class RandomCrop2D(KL.Layer):
    def __init__(self, crop_size, **kwargs):
        super(RandomCrop2D, self).__init__(**kwargs)
        self.crop_size = crop_size
        
    def get_config(self):
        config = super(RandomCrop2D, self).get_config()
        config['crop_size'] = self.crop_size
        return config
    
    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[1] = self.crop_size
        input_shape[2] = self.crop_size
        return tf.TensorShape(input_shape)
    
    def call(self, x):
        shape_x = x.get_shape()[1]
        start = tf.random.uniform([2], minval=0, maxval=shape_x - self.crop_size, dtype=tf.int32)
        y = x[:, start[0]:start[0]+self.crop_size, start[1]:start[1]+self.crop_size, :]
        y.set_shape(self.compute_output_shape(x.get_shape()))
        return y


class Compression(KL.Layer):
    def __init__(self, k: int, apply_softmax: bool, **kwargs):
        super(Compression, self).__init__(**kwargs)
        self.k = k
        self.apply_softmax = apply_softmax
        
    def get_config(self):
        config = super(Compression, self).get_config()
        config['k'] = self.k
        config['apply_softmax'] = self.apply_softmax
        
        return config
        
    
    def call(self, x):
        compressed, compressed_indices = tf.nn.top_k(x, k=self.k, sorted=False)
        
        if self.apply_softmax:
            compressed = tf.nn.softmax(compressed)
            
        compressed._name = 'compressed_values'
        compressed_indices._name = 'compressed_indices'
        
#         return tf.cast(compressed, tf.dtypes.int32), compressed_indices
        return tf.concat([compressed, tf.cast(compressed_indices, tf.float32)], axis=-1)
#         return compressed, tf.cast(compressed_indices, tf.float32)
    

# @tf.function
def decompression(x, shape, k, name):
    
    def decomp(x):
        row_tensor = tf.tile(tf.range(shape[0], dtype=tf.int32)[:, tf.newaxis], (1, k))        
        indices = tf.stack([row_tensor, tf.cast(x[..., 1:2], tf.int32)], axis=-1)
        return tf.scatter_nd(indices, x[..., :1], shape)
    
    return KL.Lambda(lambda x: decomp(x), name=name)(x)
    
        

    
class Decompression(KL.Layer):
    def __init__(self, shape, k, **kwargs):
        super(Decompression, self).__init__(**kwargs)
        self.shape = shape
        self.k_in_top_k = k
        
    def get_config(self):
        config = super(Decompression, self).get_config()
        config['shape'] = self.shape
        config['k_in_top_k'] = self.k_in_top_k
        
        return config
        
#     @tf.function(input_signature=[tf.TensorSpec(dtype=tf.dtypes.float32, shape=[None, 1]), tf.TensorSpec(dtype=tf.dtypes.int32, shape=[None, 1])])    
#     @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def call(self, x):
        
        row_tensor = tf.tile(tf.range(self.shape[0], dtype=tf.int32)[:, tf.newaxis], (1, self.k_in_top_k))        
        indices = tf.stack([row_tensor, tf.cast(x[..., :1], tf.int32)], axis=-1)
        value = tf.scatter_nd(indices, x[..., 1:2], self.shape)
#         value._name = 'decompressed_tensor'
#         utils.printd(value)
        utils.printd(value)
        utils.printd(value._type_spec)
        utils.printd(value.type_spec)
        utils.printd(dir(value))
        utils.printd(tf.keras.backend.is_keras_tensor(value))
        
        return value
        
        

    
class DummyCompressor(KL.Layer):
    def __init__(self, k):
        super(DummyCompressor, self).__init__()
        self.k = k
        
    def call(self, x):
            
        compressed, compressed_indices = tf.nn.top_k(x, k=self.k, sorted=False)
        compressed_softmax = tf.nn.softmax(compressed)
        shape_x = tf.shape(x)
        row_tensor = tf.tile(tf.range(shape_x[0])[:, tf.newaxis], (1, self.k))
        indices = tf.stack([row_tensor, compressed_indices], axis=-1)
        x = tf.scatter_nd(indices, compressed_softmax, shape_x)
        
        return x

    
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
        
        return output
        
        
class SingleOutputScaler(KL.Layer):
    def __init__(self, factor, name=None):
        super(SingleOutputScaler, self).__init__(name=name)
        self.factor = factor
        
    def call(self,tensor):

        return tensor * self.factor
        
        
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
