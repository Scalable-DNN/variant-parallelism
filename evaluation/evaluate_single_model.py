import os, gc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from glob import glob
import yaml
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

# Avoid fully occupation of gpu memory
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


import consts, utils
from dataset import Dataset


# model_paths = sorted(glob(str(consts.Paths.trained_models_dir / "*")))
model_paths = sorted(glob(str(consts.Paths.selected_models_dir / "*")))

print("="*150)
for i, path in enumerate(model_paths):
    print(f'[{i+1}]   {path}')
print("="*150)

model_path = model_paths[int(input("Which?[number]: ")) - 1]

config_name = 'train_config.yml'
with open(os.path.join(model_path, config_name), 'r') as yaml_file:
    tconf = yaml.load(yaml_file, Loader=yaml.FullLoader)


model = tf.keras.models.load_model(model_path)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, metrics=[tf.keras.metrics.TopKCategoricalAccuracy(5, name='top5'), tf.keras.metrics.TopKCategoricalAccuracy(2, name='top2'),'accuracy'])
    
# prefix = tconf['model_prefix'] % i
    
def evaluate(do_save=False):
    batch_size = 8 # tconf['train'][0]['batch'] // 4
    tfd_test = dataset.get_tfds_test(batch_size, max_n=-1)
    evs = model.evaluate(tfd_test, verbose=0)
#     msg = ''
#     msg += f' {"Ensemble":<30}{evs[-1]*100:.2f}%' + '\n'
#     for i, name in enumerate(single_model_names[::-1]):
#         msg += f' {name:<30}{evs[-(i+2)]*100:.2f}%' + '\n'
        
# #     # Save
# #     if do_save:
# #         gc.collect()
# #         with open(model_path / 'evaluation.txt', 'w') as file:
# #             file.write(msg)
# #         gc.collect()
    
#     msg = 'Evaluation Step:' + '\n' + msg[:-2]
#     utils.printd(msg, '#')
    return evs
    

rand_aug = None
if 'rand_aug' in tconf['dataset'].keys():
    rand_aug = tconf['dataset']['rand_aug']

dataset = Dataset(tconf['dataset'])

start_time = datetime.now()
evs = evaluate()
elapsed = datetime.now()

msg = ''
for i, ev in enumerate(evs[1::3]):
    msg += f'Model{i}: Top-5: {evs[1+i*3]*100:.2f}\t Top-2: {evs[2+i*3]*100:.2f}\t Top-1: {evs[3+i*3]*100:.2f}\n '
msg += 'Elapsed Time: '+''.join(str((elapsed - start_time)).split('.')[:-1])
utils.printd(msg)
# utils.printd('Elapsed Time: '+''.join(str((elapsed - start_time)).split('.')[:-1]))