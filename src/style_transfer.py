'''
reference paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
'''

import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage import io, transform
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torchvision.models import vgg19 
from torchvision import transforms

from torchsummary import summary

from util import *


def train(output_image, target_style_representation, target_content_representation, feature_maps, model, optimizer, n_steps, a, b):
    if isinstance(optimizer, optim.SGD):
        for iter in tqdm(range(n_steps)):
            # get feature maps of current output image
            _ = model(output_image)
            output_content_representation = get_content_representation(feature_maps.copy())
            output_style_representation = get_style_representation(feature_maps.copy())

            # calculate errors and backprop
            loss = a * content_loss(target_content_representation, output_content_representation) + b * style_loss(target_style_representation, output_style_representation)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    elif isinstance(optimizer, optim.LBFGS):
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            _ = model(output_image)
            output_content_representation = get_content_representation(feature_maps.copy())
            output_style_representation = get_style_representation(feature_maps.copy())

            content = content_loss(target_content_representation, output_content_representation)
            style = style_loss(target_style_representation, output_style_representation)
            loss = a * content + b * style

            if loss.requires_grad:
                loss.backward()
                
            return loss

        for iter in tqdm(range(n_steps)):
            optimizer.step(closure)
    
    return output_image


def main(style_image, content_image, n_steps, lr, optimizer, a=1e-2, b=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # for saving model later
    style_name = style_image.split('.')[0]
    content_name = content_image.split('.')[0]

    # load images and perform transformations
    style_image = read_image('../style_images/{}'.format(style_image))
    content_image = read_image('../content_images/{}'.format(content_image))

    preprocessing = transforms.Compose([transforms.Resize(512),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225]),
                                        transforms.Lambda(lambda x: x * 255)])
    postprocessing = transforms.Compose([transforms.Lambda(lambda x: x / 255),
                                        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                              std=[1/0.229, 1/0.224, 1/0.225])])            

    content_image = preprocessing(content_image).float()
    style_image = preprocessing(style_image).float()

    # get white noise image to run gradient descent on for output image
    # output_image = torch.rand(content_image.shape, device=device, requires_grad=True)
    output_image = Variable(content_image.data.clone(), requires_grad=True) # start from content image as warm start
    
    model = vgg19(pretrained=True).features # only want feature maps, don't need classifier portion
    model.to(device)
    model.eval()

    # freeze parameters so transfer learning gradient descent just affects images, doesn't retrain network
    for param in model.parameters():
        param.requires_grad = False

    # send targets to proper device
    content_image.to(device)
    style_image.to(device)

    # get style/content representations which are static for a given model/image
    feature_maps = set_hooks(model)
    _ = model(content_image)
    target_content_representation = get_content_representation({k: v.detach() for k, v in feature_maps.items()})
    _ = model(style_image)
    target_style_representation = get_style_representation({k: v.detach() for k, v in feature_maps.items()})

    if optimizer == 'sgd':
        optimizer = optim.SGD([output_image], lr=lr, momentum=0.9)
    elif optimizer == 'lbfgs':
        optimizer = optim.LBFGS([output_image], lr=lr)
    output_image = train(output_image, 
                         target_style_representation, 
                         target_content_representation, 
                         feature_maps, 
                         model, 
                         optimizer, 
                         n_steps, 
                         a, b)

    if torch.cuda.is_available():
        output_image = output_image.cpu()
    output_image = postprocessing(output_image.detach())
    output_image = output_image.numpy().squeeze()
    output_image = np.transpose(output_image, [1, 2, 0])
    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    # output_image[output_image > 1] = 1
    # output_image[output_image < 0] = 0
    plt.imsave('../output_images/{}_{}.png'.format(content_name, style_name), output_image)
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
    parser.add_argument('--lr', dest='lr', required=True,
                        help='Learning rate for optimizer')
    parser.add_argument('--optimizer', dest='optimizer', required=True,
                        help='Which pytorch optimizer to use')
    args = parser.parse_args()

    main(args.style_image, args.content_image, int(args.n_steps), float(args.lr), args.optimizer)