import os
import cv2
import time
import torch
from model import myIFCNN

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

if use_cuda:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np

from utils.myTransforms import denorm, norms, detransformcv2

fuse_scheme = 0
model_name = 'IFCNN-MAX'


# Load pretrained model
model = myIFCNN(fuse_scheme=fuse_scheme)

if use_cuda:
    model.load_state_dict(torch.load('models/' + model_name + '.pth'))
    model = model.cuda()
else:
    model.load_state_dict(torch.load('models/' + model_name + '.pth', map_location=torch.device('cpu')))
model.eval()

from utils.myDatasets import ImagePair

dataset = 'CMF'  # Color Multifocus Images
datasets_num = 20  # number of image sets in the CMF dataset
is_save = True  # if you do not want to save images, then change its value to False

testdataset_folder = 'testdataset'
output_folder = 'output'

# Determine the maximum number of digits needed
max_digits = len(str(datasets_num))

for ind in range(1, datasets_num + 1):  # Start the loop from 1
    is_gray = False  # Color (False) or Gray (True)
    mean = [0.485, 0.456, 0.406]  # normalization parameters
    std = [0.229, 0.224, 0.225]

    root = testdataset_folder
    # Use the determined maximum number of digits for the image number
    filename = 'image_{:{}}_A.jpg'.format(ind, max_digits).replace(' ', '')
    path1 = os.path.join(root, filename)
    path2 = os.path.join(output_folder, filename)

    # Load source images
    pair_loader = ImagePair(impath1=path1, impath2=path2,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)
                            ]))
    img1, img2 = pair_loader.get_pair()
    img1.unsqueeze_(0)
    img2.unsqueeze_(0)

    # Perform image fusion
    with torch.no_grad():
        if use_cuda:
            img1, img2 = img1.cuda(), img2.cuda()
        res = model(Variable(img1), Variable(img2))

        if use_cuda:
            res = res.cpu()

        res = denorm(mean, std, res[0]).clamp(0, 1) * 255
        res_img = res.numpy().astype('uint8')
        img = res_img.transpose([1, 2, 0])

    # Save fused images
    if is_save:
        filename = model_name + '-' + dataset + '-' + 'image_{:0{}}_A'.format(ind, max_digits)
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = Image.fromarray(img)
            img.save('results/' + filename + '.png', format='PNG', compress_level=0)
        else:
            img = Image.fromarray(img)
            img.save('results/' + filename + '.png', format='PNG', compress_level=0)

# when evaluating time costs, remember to stop writing images by setting is_save = False
