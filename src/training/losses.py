from tensorflow.keras import losses as tf_losses

import utils



class Losses:
    
    def __init__(self, 
                 single_outputs_coeffs, 
                 final_outputs_coeffs,
                 single_outputs_names: str=None, 
                 final_outputs_names: str=None):
        
        self.single_outputs_coeffs = single_outputs_coeffs
        self.final_outputs_coeffs = final_outputs_coeffs
        self.single_outputs_names = single_outputs_names
        self.final_outputs_names = final_outputs_names
        self.losses = {}
        self.loss_weights = {}
        self.cat_ce_loss = tf_losses.CategoricalCrossentropy
        
    def add_loss_to_dict(self, key, coeff):
        self.losses[key] = self.cat_ce_loss()
        self.loss_weights[key] = coeff
    
    def get_loss_dict(self):
        '''Returns two dictinonaries containing 1) loss functions and their names and 2) weights and their names.'''
        
        if isinstance(self.final_outputs_names, list):
            for loss_name, coeff in zip(self.final_outputs_names, self.final_outputs_coeffs):
                self.add_loss_to_dict(loss_name, coeff)
        else:
            loss_name = self.final_outputs_names
            self.add_loss_to_dict(loss_name, self.single_outputs_coeffs)
        
        for loss_name, coeff in zip(self.single_outputs_names, self.single_outputs_coeffs):
            self.add_loss_to_dict(loss_name, coeff)
        
        return self.losses, self.loss_weights
            

    
    
# def get_loss(y_true, y_pred, target_class, loss_type='binary'):

#     def multi_class_loss():
#         target_true = y_true[..., (target_class*5):(target_class+1)*5]
# #         binary_target_true = tf.argmax(target_true, axis=-1)
# #         return tf.keras.losses.binary_crossentropy(binary_target_true, y_pred)
#         return tf_losses.categorical_crossentropy(target_true, y_pred)

#     def binary_loss():
#         target_true = y_true[..., (target_class):(target_class+1)]
#         return tf_losses.binary_crossentropy(target_true, y_pred)

#     return binary_loss() if loss_type == 'binary' else multi_class_loss()


# def get_losses(coeffs, alpha: list):
#     losses = {}
#     loss_weights = {}
#     sum_w_losses = 0
#     for i in range(len(input_shapes)):
#         pred_head_name = tconf['model_prefix'] % i + 'single_pred'
#         losses[pred_head_name] = tf_losses.CategoricalCrossentropy()
#         loss_weights[pred_head_name] = coeffs[0]
# #         w_loss = tconf['input_shapes'][i] / max(tconf['input_shapes']) * (alpha[i] / .35)
# #         sum_w_losses += w_loss
# #         loss_weights[pred_head_name] = w_loss

#     losses['final_prediction'] = tf_losses.CategoricalCrossentropy()
#     loss_weights['final_prediction'] = coeffs[1]
# #     loss_weights['final_prediction'] = sum_w_losses * 2
    
#     utils.printd(f'Loss Weights: {loss_weights}')

#     return losses, loss_weights
