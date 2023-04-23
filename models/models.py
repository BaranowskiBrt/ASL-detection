from .gru import GruModel
from .linear import LinearModel, LinearSplitModel
from .transformer import TransformerModel


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024**2


def choose_model(model_type, num_classes, keypoints_len, frame_len, dropout_p):
    if model_type == "gru":
        model = GruModel(
            num_classes=num_classes,
            keypoints_len=keypoints_len,
        )
    elif model_type == "linear":
        model = LinearModel(
            num_classes=num_classes,
            keypoints_len=keypoints_len,
            frame_len=frame_len,
            dropout_p=dropout_p,
        )
    elif model_type == "linear_split":
        model = LinearSplitModel(
            num_classes=num_classes,
            keypoints_len=keypoints_len,
            frame_len=frame_len,
            dropout_p=dropout_p,
        )
    elif model_type == "transformer":
        model = TransformerModel(
            num_classes=num_classes,
            keypoints_len=keypoints_len,
            max_frame_len=frame_len,
            dropout_p=dropout_p,
        )
    print(f"Model size: {get_model_size(model):.2f}MB")
    return model
