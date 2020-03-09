# -*-coding:utf-8 -*-
"""
    DataSet class
"""
import os
import torch
import numpy as np
from PIL import Image
from PIL import ImageFilter
from utils import get_soft_label
from torchvision import transforms
from utils import get_label_from_txt, get_info_from_txt
from torch.utils.data import DataLoader
from utils import get_attention_vector, get_vectors
from torch.utils.data.dataset import Dataset
from utils import Vector300W, Bbox300W

torch.manual_seed(0)

def loadData(data_dir, input_size, batch_size, num_classes, training=True):
    """

    :return:
    """
    # define transformation
    if training:
        transformations = transforms.Compose([transforms.Resize((input_size, input_size)),
                                              #transforms.RandomCrop(input_size), # delete(will cause object translation)
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = TrainDataSet(data_dir, transformations, num_classes)
        print("Traning sampels:", train_dataset.length)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader
    else:
        transformations = transforms.Compose([transforms.Resize((input_size, input_size)),
                                              #transforms.RandomCrop(input_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_dataset = TestDataSet(data_dir, transformations, num_classes)

        # initialize train DataLoader
        data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        return data_loader


class TrainDataSet(Dataset):
    def __init__(self, data_dir, transform, num_classes, image_mode="RGB"):
        self.data_dir = data_dir
        self.transform = transform
        self.num_classes = num_classes
        self.image_mode = image_mode
        self.bins = np.linspace(-1,1,self.num_classes) 
        self.bins = np.linspace(-1,1,self.num_classes)
        self.slop = 1.0

        self.data_list = os.listdir(os.path.join(self.data_dir, 'imgs'))
        self.length = len(self.data_list)

    def __getitem__(self, index, crop=True):
        # data basename
        base_name, _ = self.data_list[index].split('.')

        # read image file
        img = Image.open(os.path.join(self.data_dir, "imgs/" + base_name + ".jpg"))
        img = img.convert(self.image_mode)

        if crop:

            # get face bounding box
            bbox = Bbox300W(os.path.join(self.data_dir, "labels/" + base_name + ".txt"))
            x_min, y_min, x_max, y_max = bbox

            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Augmentation:Blur?
        if np.random.random_sample() < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Augmentation:Gray?
        if np.random.random_sample():
            img = img.convert('L').convert("RGB")

        # transform
        if self.transform:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]


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

        return img, soft_label_l, soft_label_d, soft_label_f, vector_label_l, vector_label_d, vector_label_f, os.path.join(self.data_dir, "imgs/" + base_name + ".jpg")

    def __len__(self):
        return self.length


class TestDataSet(Dataset):
    def __init__(self, data_dir, transform, num_classes, image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.num_classes = num_classes

        self.data_list = os.listdir(os.path.join(self.data_dir, 'imgs'))

        self.image_mode = image_mode
        self.length = len(self.data_list)

    def __getitem__(self, index, crop=True):
        base_name, _ = self.data_list[index].split('.')
        img = Image.open(os.path.join(self.data_dir, 'imgs/' + base_name + '.jpg'))
        img = img.convert(self.image_mode)

        if crop:
            # get face bbox
            bbox = Bbox300W(os.path.join(self.data_dir, 'labels/' + base_name + '.txt'))
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]

            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))


        if self.transform is not None:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        #get left vector, down vector and front vector
        left_vector, down_vector, front_vector  = Vector300W(os.path.join(self.data_dir, "labels/" + base_name + '.txt'))

        vector_label_l = torch.FloatTensor(left_vector)
        vector_label_d = torch.FloatTensor(down_vector)
        vector_label_f = torch.FloatTensor(front_vector)

        bins = np.linspace(-1,1,self.num_classes)

        #----------------left vector-------------------------
        # classification label
        classify_label = torch.LongTensor(np.digitize(left_vector, bins)) # return the index
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label_l = torch.stack([soft_label_x, soft_label_y, soft_label_z])

        #-------------------down vector--------------
        classify_label = torch.LongTensor(np.digitize(down_vector, bins)) # return the index
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label_d = torch.stack([soft_label_x, soft_label_y, soft_label_z])


        #------------------front vector-------------------
        classify_label = torch.LongTensor(np.digitize(front_vector, bins)) # return the index
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label_f = torch.stack([soft_label_x, soft_label_y, soft_label_z])

        return img, soft_label_l, soft_label_d, soft_label_f, vector_label_l, vector_label_d, vector_label_f, os.path.join(self.data_dir, "imgs/" + base_name + ".jpg")

    def __len__(self):
        # 1,969
        return self.length
