# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES=1 #,0
# export TF_XLA_FLAGS='--tf_xla_enable_xla_devices'
export AUTOGRAPH_VERBOSITY=0
#
#
#
#
# cd "."
python multiplenets.py \
  --config='train_config_ensemble_imnet.yml'