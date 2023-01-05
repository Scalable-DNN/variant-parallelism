import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf

from dataset import Dataset
from model import NetworkConstructor
import consts, utils
from losses import Losses
import custom_layers as CL


if __name__ == '__main__':
    
    # Create tflite directory if it does not exist.
    tflites_dir = consts.Paths.tflite_models_dir
    utils.check_create_path(tflites_dir)

    # Select TF model(s)
#     models_dir = consts.Paths.selected_models_dir
    models_dir = consts.Paths.detached_models_dir
    model_path = utils.list_select_models(models_dir)
    
    if isinstance(model_path, str):
        model_path = [model_path]
    
    for mpath in model_path:
        
        # Loading model ...
        utils.printd(f'Loading model {Path(mpath).name}...')
        
        model = tf.keras.models.load_model(mpath)
        
        # Correct model
        model_x = tf.keras.Model(model.input, model.layers[0].output)
        
        x = model.layers[1](model_x.output)
        
        for i in range(2, len(model.layers)):
            x = model.layers[i](x)
        
        model_x = tf.keras.Model(model.input, x, name=model.name)
        
        # Convert to TFLite
        utils.printd(f'Converting model {Path(mpath).name}...')
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model_x)

        converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
          tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        
#         converter.allow_custom_ops = True

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8  # or tf.uint8
        # converter.inference_output_type = tf.int8  # or tf.uint8
        
        tflite_model = converter.convert()

        # Save the model.
        utils.printd(f'Saving model {Path(mpath).name}...')
        
        model_name = Path(mpath).name + ".tflite"
        tflite_path = str(tflites_dir / model_name)

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
    utils.printd("Done.")