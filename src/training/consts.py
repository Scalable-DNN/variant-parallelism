from pathlib import Path


class Paths:
    ROOT = Path("/opt/project/data")
    class_paral = ROOT #/ "trained_models"
    
#     ds_dir = CIFAR_DIR / "datasets"
#     train_ds = ds_dir / "train_ds.npy"
#     test_ds = ds_dir / "test_ds.npy"
    
#     weights_dir = CIFAR_DIR / "models"
#     pretrained_weights_dir = weights_dir / "pretrained_models_notop"
    
#     predictions_dir = ROOT / "predictions"
    trained_models_dir = class_paral / "trained_models"
    selected_models_dir = trained_models_dir / "selected"
    detached_models_dir = selected_models_dir / "detached"
    tflite_models_dir = detached_models_dir / "tflite_models"