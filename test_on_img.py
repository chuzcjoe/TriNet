# -*- coding:utf-8 -*-
import os
import tqdm
import utils
import torch
import argparse
import cv2 as cv
import numpy as np
import torch.nn as nn
from dataset import loadData
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms
import matplotlib.pyplot as plt
from net import ResNet, MobileNetV2
import torchvision
from Optimization.optimize import Optimize

class Test:

    def __init__(self, model_name, snapshot, num_classes):
        
        self.num_classes = num_classes
        if model_name == "resnet50":
            self.model = ResNet(torchvision.models.resnet50(pretrained=False), num_classes)
        elif model_name == "mobilenetv2":
            self.model = MobileNetV2(num_classes=num_classes)
        else:
           print("No such model...")
           exit(0)

        self.saved_state_dict = torch.load(snapshot)
        self.model.load_state_dict(self.saved_state_dict)
        self.model.cuda(0)
        self.model.eval()  # Change model to 'eval' mode
        
        self.softmax = nn.Softmax(dim=1).cuda(0)

        self.optimizer = Optimize()

    def draw_vectors(self, pred_vector1, pred_vector2, pred_vector3, img, center, width):

        
        optimize_v = self.optimizer.Get_Ortho_Vectors(np.array(pred_vector1), np.array(pred_vector2), np.array(pred_vector3))
        v1, v2, v3 = optimize_v[0], optimize_v[1], optimize_v[2]

        # draw vector in blue color
        predx, predy, predz = v1
        utils.draw_front(img, predx, predy, width, tdx=center[0], tdy=center[1], size=100, color=(255, 0, 0))

        # draw vector in green color
        predx, predy, predz = v2
        utils.draw_front(img, predx, predy, width, tdx=center[0], tdy=center[1], size=100, color=(0, 255, 0))

        # draw vector in red color
        predx, predy, predz = v3
        utils.draw_front(img, predx, predy, width, tdx=center[0], tdy=center[1], size=100, color=(0, 0, 255))
        
        cv.imshow("pose visualization",img)

    def test_per_img(self, cv_img, draw_img, center, w):
        with torch.no_grad():
            images = cv_img.cuda(0)

            # get x,y,z cls predictions
            x_v1, y_v1, z_v1, x_v2, y_v2, z_v2, x_v3, y_v3, z_v3 = self.model(images)

            # get prediction vector(get continue value from classify result)
            _, _, _, pred_vector1 = utils.classify2vector(x_v1, y_v1, z_v1, self.softmax, self.num_classes)
            _, _, _, pred_vector2 = utils.classify2vector(x_v2, y_v2, z_v2, self.softmax, self.num_classes)
            _, _, _, pred_vector3 = utils.classify2vector(x_v3, y_v3, z_v3, self.softmax, self.num_classes)

            #visualize vectors
            self.draw_vectors(pred_vector1[0].cpu().tolist(), pred_vector2[0].cpu().tolist(), pred_vector3[0].cpu().tolist(),
                                          draw_img, center, w)
