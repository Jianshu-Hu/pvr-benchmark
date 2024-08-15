import timm
import os


def load_encoder(model_type, checkpoint_path=None):
    if checkpoint_path is not None:
        if not os.path.isabs(checkpoint_path):
            model_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
            checkpoint_path = os.path.join(model_base_dir, checkpoint_path)
        model = timm.create_model(model_type, pretrained=True, num_classes=0,
                                  pretrained_cfg={'file': checkpoint_path})
    else:
        model = timm.create_model(model_type, pretrained=True, num_classes=0)
    model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return model, transforms
