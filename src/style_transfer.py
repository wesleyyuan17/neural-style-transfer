'''
reference paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
'''

import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage import io, transform

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19 
from torchsummary import summary


# INPUT_SIZE = (3, 224, 224)
FEATURE_MAP_IDX = {3: 'fm1', 8: 'fm2', 17: 'fm3', 26: 'fm4', 35: 'fm5'} # output of ReLU layers right before 5 pooling layers


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
    feature_maps = get_content_representation(feature_maps)
    for k, v in feature_maps.items():
        feature_maps[k] = (torch.matmul(v, v.t()), v.shape[1])
    
    return feature_maps


def get_content_representation(feature_maps):
    # per paper, flatten to have n_filter x (height * width) representation
    for k, v in feature_maps.items():
        feature_maps[k] = v.flatten(start_dim=1)

    return feature_maps


def style_loss(target, output):
    total_loss = 0
    for k in target.keys():
        G, M = target[k]
        N = G.shape[0]
        A, _ = output[k]
        E = torch.sum((G - A)**2) / (4 * N * N * M * M)
        total_loss += 0.2 * E
    
    return total_loss


def content_loss(target, output):
    P = target['fm4']
    F = output['fm4']
    return 0.5 * torch.sum((F - P)**2)


def train(output_image, target_style_representation, target_content_representation, feature_maps, model, optimizer, n_steps, a, b):
    for iter in range(n_steps):
        # get feature maps of current output image
        _ = model(output_image)
        output_content_representation = get_content_representation(feature_maps.copy())
        output_style_representation = get_style_representation(feature_maps.copy())

        # calculate errors and backprop
        loss = a * content_loss(target_content_representation, output_content_representation) + b * style_loss(target_style_representation, output_style_representation)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return output_image


def main(style_image, content_image, n_steps, a=0.1, b=1e-3):
    # load images and models
    style_image = read_image('../style_images/{}'.format(style_image))
    content_image = torch.Tensor(read_image('../content_images/{}'.format(content_image))).permute([2, 0, 1])
    style_image = torch.Tensor(transform.resize(style_image, content_image.shape))

    # get white noise image to run gradient descent on for output image
    output_image = torch.rand(content_image.shape, requires_grad=True)

    model = vgg19(pretrained=True).features # only want feature maps, don't need classifier portion

    # freeze parameters so transfer learning gradient descent just affects images, doesn't retrain network
    for param in model.parameters():
        param.requires_grad = False

    # summary(model, INPUT_SIZE)
    # print(model)

    # get style/content representations which are static for a given model/image
    feature_maps = set_hooks(model)
    _ = model(content_image)
    target_content_representation = get_content_representation({k: v.detach() for k, v in feature_maps.items()})
    _ = model(style_image)
    target_style_representation = get_style_representation({k: v.detach() for k, v in feature_maps.items()})

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    output_image = train(output_image, 
                         target_style_representation, 
                         target_content_representation, 
                         feature_maps, 
                         model, 
                         optimizer, 
                         n_steps, 
                         a, b)

    output_image = output_image.detach().numpy()
    output_image = np.transpose(output_image, [1, 2, 0])
    plt.imsave('../output_images/test.png', output_image)
    plt.imshow(output_image)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_image', dest='style_image', required=True,
                        help='The path image from which style of new image should be taken')
    parser.add_argument('--content_image', dest='content_image', required=True,
                        help='The path image from which content of new image should be taken')
    parser.add_argument('--n_steps', dest='n_steps', required=True,
                        help='The number of training steps to take in generating final image')
    args = parser.parse_args()

    main(args.style_image, args.content_image, int(args.n_steps))