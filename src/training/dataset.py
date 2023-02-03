import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ['0', '1', '-1'][1]

import tensorflow as tf
import tensorflow_datasets as tfds
# from official.vision.image_classification.augment import RandAugment


class Dataset:

    def __init__(self, dataset_config, batch=512, shuffle_size=None):
        self.dataset = dataset_config['name']
        
        self.rand_aug = None
        # if 'rand_aug' in dataset_config.keys():
        #     params = dataset_config['rand_aug']
        #     self.rand_aug = RandAugment(num_layers=params['n'], magnitude=params['m'])
        #     print("RANDOM AUGMENTATION IS ENABLED.")
        
#         self.datasets = {
#             'cifar10': tf.keras.datasets.cifar10.load_data,
#             'cifar100': tf.keras.datasets.cifar100.load_data,
#             'food101': tfds.load('food101', split=['train', 'validation'], shuffle_files=True, as_supervised=True, with_info=True),
#             # TODO: implement the following datasets
#             'imagenet2012': None,  # tfds.load('imagenet2012'split=['train', 'validation'], shuffle_files=True, as_supervised=True) 
#             'flowers': None,
#             'svhn': None,
#         }
        self.dataset_config = dataset_config

#         (self.x_train, self.y_train), (self.x_test, self.y_test) = \
#             self.datasets[self.dataset]()

#         self.y_test = tf.keras.utils.to_categorical(self.y_test)
#         self.y_train = tf.keras.utils.to_categorical(self.y_train)

        self.batch, self.shuffle = batch, shuffle_size
        if self.shuffle is None:
            self.shuffle = min(max(self.batch * 4, 256), 1024)
            
        self.AUTO = tf.data.AUTOTUNE
            
    def apply_dataset_specific_map(self):
        if self.dataset in ['food101', 'stanford_dogs', 'imagenet2012']:
#             return True, lambda x, y: (tf.image.resize(x, (96, 96), 'bilinear'), y)
#             return True, lambda x, y: (tf.image.resize(x, (234, 234), 'bilinear'), y)
            return True, lambda x, y: (tf.image.resize(x, self.dataset_config['network_input_size'][:2], 'bilinear'), y)
        
        return False, None
        

    def get_tfds_test(self, batch=None, drop_remainder=False, max_n=8000):
        split = 'test'
        batch = self.batch if batch is None else batch
        try:
            tfd_test, ds_info = tfds.load(self.dataset, split=split, as_supervised=True, with_info=True)
        except:
            split = 'validation'
            tfd_test, ds_info = tfds.load(self.dataset, split=split, as_supervised=True, with_info=True)
                        
        cond, map_func = self.apply_dataset_specific_map()
        if cond:
            tfd_test = tfd_test.map(map_func, num_parallel_calls=self.AUTO)
        
        
#         if max_n > 0:
#             tfd_test = tfd_test.take(max_n)
        
        num_classes = ds_info.features['label'].num_classes
        tfd_test = (tfd_test
            .batch(batch, drop_remainder=drop_remainder)
            .map(lambda x,y: (x, tf.one_hot(y, num_classes)), num_parallel_calls=self.AUTO)
            .prefetch(self.AUTO))

        return tfd_test

    def get_tfds_train(self, batch=None, shuffle=None, flag_called_from_mixup=False):
        
        
        # Check if the method was called from the mixup method, the statement must not run.
        if self.dataset_config['type'] == 'mixup' and not flag_called_from_mixup:
            return self.mix_up(batch, alpha=self.dataset_config['alpha'])
        
        
        batch = self.batch if batch is None else batch
        shuffle = self.shuffle if shuffle is None else shuffle
#         tfd_train = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(shuffle)
        if self.dataset in ['imagenet2012']:
            tfd_train, ds_info = tfds.load(self.dataset, split='train', shuffle_files=True, as_supervised=True, with_info=True)
        else:
            tfd_train, ds_info = tfds.load(self.dataset, split='train', shuffle_files=True, as_supervised=True, with_info=True)
    
        num_classes = ds_info.features['label'].num_classes

        if self.dataset not in ['imagenet2012']:
            print("Data Shuffling is enabled.")
            tfd_train = tfd_train.shuffle(shuffle)
                    
        if self.rand_aug:
            tfd_train = tfd_train.map(
                lambda x, y: (self.rand_aug.distort(x), y), self.AUTO)
        
        cond, map_func = self.apply_dataset_specific_map()
        if cond:
            tfd_train = tfd_train.map(map_func, num_parallel_calls=self.AUTO)
            
        tfd_train = (tfd_train
             .batch(batch, drop_remainder=True)
             .map(lambda x, y: (x, tf.one_hot(y, num_classes)), num_parallel_calls=self.AUTO)
             .prefetch(self.AUTO))
        
        return tfd_train

    # majority of the MixUP implementation was adapted from
    # https://keras.io/examples/vision/mixup/
    def sample_beta_distribution(self, size, concentration_0=.2,
                                 concentration_1=.2):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        self.beta = gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    @tf.function
    def mixup_func(self, ds_0, ds_1):
        images_0, labels_0 = ds_0
        images_1, labels_1 = ds_1
        batch_size = tf.shape(images_0)[0]

        # Sample lambda and reshape it to do the mixup
#         lmbda = self.sample_beta_distribution(batch_size, self.alpha, self.alpha)
        x_l = tf.reshape(self.beta, (batch_size, 1, 1, 1))
        y_l = tf.reshape(self.beta, (batch_size, 1))

        images_0 = tf.cast(images_0, tf.float32)
        images_1 = tf.cast(images_1, tf.float32)
        # Perform mixup on both images and labels by combining a pair of
        # images/labels (one from each dataset) into one image/label
        images = images_0 * x_l + images_1 * (1 - x_l)
        labels = labels_0 * y_l + labels_1 * (1 - y_l)
        return (images, labels)

    def mix_up(self, batch=None, shuffle=None, alpha=0.2):
        
        print("MIXUP AUGMENTATION IS ENABLED.")
        
        self.alpha = alpha
        self.sample_beta_distribution(batch, self.alpha, self.alpha)
        train_ds_0 = self.get_tfds_train(batch=batch, shuffle=shuffle, flag_called_from_mixup=True)
        train_ds_1 = self.get_tfds_train(batch=batch, shuffle=shuffle, flag_called_from_mixup=True)
        train_ds_mu = tf.data.Dataset.zip((train_ds_0, train_ds_1))
        train_ds_mu = train_ds_mu.map(
            lambda ds_0, ds_1: self.mixup_func(ds_0, ds_1), num_parallel_calls=self.AUTO)

        return train_ds_mu