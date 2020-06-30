# -*-coding:utf-8 -*-
"""
    DataSet class
"""
import os
import logging
logging.disable(logging.WARNING)
import torch
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter
from utils import get_soft_label
from torchvision import transforms
from utils import get_label_from_txt, get_info_from_txt
from torch.utils.data import DataLoader
from utils import get_attention_vector, get_vectors
from torch.utils.data.dataset import Dataset
from utils import *
import tensorflow as tf

torch.manual_seed(0)

def random_crop(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]
    out = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    out = cv2.resize(out, (h,w), interpolation=cv2.INTER_CUBIC)
    return out

def random_crop_black(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out

def random_crop_white(x,dn):
    dx = np.random.randint(dn,size=1)[0]
    dy = np.random.randint(dn,size=1)[0]
    h = x.shape[0]
    w = x.shape[1]

    dx_shift = np.random.randint(dn,size=1)[0]
    dy_shift = np.random.randint(dn,size=1)[0]
    out = x*0+255
    out[0+dy_shift:h-(dn-dy_shift),0+dx_shift:w-(dn-dx_shift),:] = x[0+dy:h-(dn-dy),0+dx:w-(dn-dx),:]
    
    return out


def random_cutout(img, patches, size):
    h = img.shape[0]
    w = img.shape[1]

    for i in range(patches):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)

        img[y1: y2, x1: x2, :] = 0

    return img


def random_blur(img, kernel_size):
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    return img

def random_hue(image, saturations=[5,10,15,20]):
    saturation = saturations[np.random.randint(4)]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = image[:,:,2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)

    image[:,:,2] = v
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def random_light(image, gammas=[0.1,0.3,0.5,0.7,1.0,1.5,2.0]):
    gamma = gammas[np.random.randint(7)]

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    image=cv2.LUT(image, table)

    return image

def augment_data(image):
    rand_r = np.random.random()
    rand_l = np.random.random()

    if rand_l < 0.3:
        image = random_light(image)


    if  rand_r < 0.25:
        dn = np.random.randint(15,size=1)[0]+1
        image = random_crop(image,dn)
    
    elif rand_r >= 0.25 and rand_r < 0.5:
        dn = np.random.randint(15,size=1)[0]+1
        image = random_crop_black(image,dn)
    
    elif rand_r >= 0.5 and rand_r < 0.75:
        dn = np.random.randint(15,size=1)[0]+1
        image = random_crop_white(image,dn)

    else:
        image = random_cutout(image, 2, 10)
    
    if np.random.random() > 0.3:
        image = tf.contrib.keras.preprocessing.image.random_zoom(image, [0.8,1.2], row_axis=0, col_axis=1, channel_axis=2)
        
    return image


def loadData(data_dir, input_size, batch_size, num_classes, training=True):

    if training:
        train_dataset = TrainDataSet(data_dir, input_size, num_classes)
        print("Traning sampels:", train_dataset.length)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader
    else:
        test_dataset = TestDataSet(data_dir, input_size, num_classes)
        data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        return data_loader


class TrainDataSet(Dataset):
    def __init__(self, data_dir, input_size, num_classes, crop=True):
        self.data_dir = data_dir
        self.input_size = input_size
        self.num_classes = num_classes

        self.bins = np.linspace(-1,1,self.num_classes) 
        self.crop = crop

        self.data_list = os.listdir(os.path.join(self.data_dir, 'imgs'))
        self.length = len(self.data_list)

    def __getitem__(self, index):

        base_name, _ = self.data_list[index].split('.')
        # read image file
        img = cv2.imread(os.path.join(self.data_dir, "imgs/" + base_name + ".jpg"))

        #crop head part
        if self.crop:
            # get face bounding box
            bbox = bbox_300W(os.path.join(self.data_dir, "labels/" + base_name + ".txt"))
            x_min, y_min, x_max, y_max = bbox
            img = img[y_min:y_max, x_min:x_max]
        
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = augment_data(img)

        #C,H,W to H,W,C
        img = img.swapaxes(1,2).swapaxes(0,1)
        img = np.ascontiguousarray(img)


        #get left vector, down vector, front vector
        left_vector, down_vector, front_vector  = Vector300W(os.path.join(self.data_dir, "labels/" + base_name + '.txt'))

        vector_label_l = torch.FloatTensor(left_vector)
        vector_label_d = torch.FloatTensor(down_vector)
        vector_label_f = torch.FloatTensor(front_vector)

        #----------------left vector-------------------------
        # classification label
        classify_label = torch.LongTensor(np.digitize(left_vector, self.bins)) # return the index
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label_l = torch.stack([soft_label_x, soft_label_y, soft_label_z])

        #-------------------down vector--------------
        classify_label = torch.LongTensor(np.digitize(down_vector, self.bins)) # return the index
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label_d = torch.stack([soft_label_x, soft_label_y, soft_label_z])


        #------------------front vector-------------------
        classify_label = torch.LongTensor(np.digitize(front_vector, self.bins)) # return the index
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label_f = torch.stack([soft_label_x, soft_label_y, soft_label_z])

        return torch.from_numpy(img), soft_label_l, soft_label_d, soft_label_f, vector_label_l, vector_label_d, vector_label_f, os.path.join(self.data_dir, "imgs/" + base_name + ".jpg"), vector_label_l, vector_label_d, vector_label_f

    def __len__(self):
        return self.length


class TestDataSet(Dataset):
    def __init__(self, data_dir, input_size, num_classes, crop=True):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.crop = crop
        self.input_size= input_size
        self.bins = np.linspace(-1,1,self.num_classes)
        self.data_list = os.listdir(os.path.join(self.data_dir, 'imgs'))
        self.length = len(self.data_list)

    def __getitem__(self, index):
        base_name, _ = self.data_list[index].split('.')
        img = cv2.imread(os.path.join(self.data_dir, "imgs/" + base_name + ".jpg"))

        if self.crop:
            # get face bbox
            bbox = bbox_300W(os.path.join(self.data_dir, 'labels/' + base_name + '.txt'))
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]
            img = img[y_min:y_max, x_min:x_max]

        img = cv2.resize(img, (self.input_size, self.input_size))

        img = img.swapaxes(1,2).swapaxes(0,1)
        img = np.ascontiguousarray(img)


        #get left vector, down vector and front vector
        left_vector, down_vector, front_vector  = Vector300W(os.path.join(self.data_dir, "labels/" + base_name + '.txt'))

        vector_label_l = torch.FloatTensor(left_vector)
        vector_label_d = torch.FloatTensor(down_vector)
        vector_label_f = torch.FloatTensor(front_vector)

        #----------------left vector-------------------------
        # classification label
        classify_label = torch.LongTensor(np.digitize(left_vector, self.bins)) # return the index
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label_l = torch.stack([soft_label_x, soft_label_y, soft_label_z])

        #-------------------down vector--------------
        classify_label = torch.LongTensor(np.digitize(down_vector, self.bins)) # return the index
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label_d = torch.stack([soft_label_x, soft_label_y, soft_label_z])


        #------------------front vector-------------------
        classify_label = torch.LongTensor(np.digitize(front_vector, self.bins)) # return the index
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label_f = torch.stack([soft_label_x, soft_label_y, soft_label_z])

        return torch.from_numpy(img), soft_label_l, soft_label_d, soft_label_f, vector_label_l, vector_label_d, vector_label_f, os.path.join(self.data_dir, "imgs/" + base_name + ".jpg"), vector_label_l, vector_label_d, vector_label_f

    def __len__(self):
        # 1,969
        return self.length
