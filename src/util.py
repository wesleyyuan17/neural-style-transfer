from skimage import io

import torch


# what I think are the layers conv1_1, conv2_1, conv3_1, conv4_1, conv4_2, conv5_1
FEATURE_MAP_IDX = {1: 'fm1', 6: 'fm2', 11: 'fm3', 20: 'fm4', 22: 'content', 29: 'fm5'}


def read_image(path):
    return io.imread(path)


def set_hooks(model):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    for k, v in FEATURE_MAP_IDX.items():
        model[k].register_forward_hook(get_activation(v))
    
    return activation


def get_style_representation(feature_maps):
    # per paper, after flatten inner product b/w rows of vectorized feature maps
    del feature_maps['content']
    for k, v in feature_maps.items():
        v = v.flatten(start_dim=1)
        feature_maps[k] = torch.matmul(v, v.t()) / (v.shape[0] * v.shape[1])
    
    return feature_maps


def get_content_representation(feature_maps):
    # per paper, flatten to have n_filter x (height * width) representation
    return feature_maps['content'].flatten(start_dim=1)


def style_loss(target, output):
    total_loss = 0
    for k in target.keys():
        G = target[k]
        A = output[k]
        E = 0.25 * torch.sum((G - A)**2)
        total_loss += 0.2 * E
    
    return total_loss


def content_loss(target, output):
    return 0.5 * torch.sum((target - output)**2)