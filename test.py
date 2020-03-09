# -*- coding:utf-8 -*-
import os
import tqdm
import utils
import torch
import argparse
import cv2 as cv
import numpy as np
import torch.nn as nn
from net import MobileNetV2, ResNet
from dataset import loadData
import pickle
import torchvision

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--test_data', dest='test_data', help='Directory path for data.',
                        default='', type=str)
    parser.add_argument("--model_name", dest="model_name",type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=64, type=int)
    parser.add_argument('--degree_error_limit', dest='degree_error_limit', help='degrees error for calc cs',
                        default=10, type=int)
    parser.add_argument('--save_dir', dest='save_dir', help='directory for saving drawn pic',
                        default='./plot', type=str)
    parser.add_argument('--show_front', dest='show_front', help='show front or not',
                        default=False, type=bool)
    parser.add_argument('--analysis', dest='analysis', help='analysis result or not',
                        default=False, type=bool)
    parser.add_argument('--collect_score', dest='collect_score', help='show huge error or not',
                        default=False, type=bool)
    parser.add_argument('--num_classes', dest='num_classes', help='number of classify',
                        default=90, type=int)
    parser.add_argument('--width_mult', dest='width_mult', choices=[0.5, 1.0], help='mobilenet_v2 width_mult',
                        default=1.0, type=float)
    parser.add_argument('--input_size', dest='input_size', choices=[224, 192, 160, 128, 96], help='size of input images',
                        default=224, type=int)
    parser.add_argument('--write_error', dest='write_error', choices=[224, 192, 160, 128, 96], help='size of input images',
                        default=False, type=bool)
    parser.add_argument('--write_vector', dest='write_vector', help='write predicted vectors to local txt files',
                        default=True, type=bool)
    args = parser.parse_args()
    return args


def draw_attention_vector(vector_label, pred_vector, img_path, args):
    #save_dir = os.path.join(args.save_dir, 'show_front')
    img_name = os.path.basename(img_path)

    img = cv.imread(img_path)

    predx, predy, predz = pred_vector

    # draw pred attention vector with red
    utils.draw_front(img, predy, predz, tdx=None, tdy=None, size=100, color=(0, 0, 255))

    #cv.imwrite(os.path.join(save_dir, img_name), img)


def test(model, test_loader, softmax, args):
    if args.analysis:
        utils.mkdir(os.path.join(args.save_dir, 'analysis'))
        loss_dict = {'img_name': list(),'degree_error_f': list(),'degree_error_r': list(), 'degree_error_u':list()}

    if args.write_error:
        error_5 = {'img_name':list(), 'degree_error': list()}

    l_total_err = 0.0
    d_total_err = 0.0
    f_total_err = 0.0

    total = 0.0
    score = 0.0
    for i, (images, cls_label_f, cls_label_r, cls_label_u, vector_label_f, vector_label_r, vector_label_u, names) in enumerate(tqdm.tqdm(test_loader)):
        with torch.no_grad():
            #print(images.shape)
            images = images.cuda(0)

            vector_label_f = vector_label_f.cuda(0)
            vector_label_r = vector_label_r.cuda(0)
            vector_label_u = vector_label_u.cuda(0)

            # get x,y,z cls predictions
            x_cls_pred_f, y_cls_pred_f, z_cls_pred_f,x_cls_pred_r, y_cls_pred_r, z_cls_pred_r,x_cls_pred_u, y_cls_pred_u, z_cls_pred_u = model(images)

            # get prediction vector(get continue value from classify result)
            _, _, _, pred_vector_f = utils.classify2vector(x_cls_pred_f, y_cls_pred_f, z_cls_pred_f, softmax, args.num_classes, )

            _, _, _, pred_vector_r = utils.classify2vector(x_cls_pred_r, y_cls_pred_r, z_cls_pred_r, softmax, args.num_classes, )

            _, _, _, pred_vector_u = utils.classify2vector(x_cls_pred_u, y_cls_pred_u, z_cls_pred_u, softmax, args.num_classes, )

            # Mean absolute error
            cos_value_f = utils.vector_cos(pred_vector_f, vector_label_f)
            degrees_error_f = torch.acos(cos_value_f) * 180 / np.pi
            #print(degrees_error_f)

            cos_value_r = utils.vector_cos(pred_vector_r, vector_label_r)
            degrees_error_r = torch.acos(cos_value_r) * 180 / np.pi

            cos_value_u = utils.vector_cos(pred_vector_u, vector_label_u)
            degrees_error_u = torch.acos(cos_value_u) * 180 / np.pi

            l_total_err += torch.mean(degrees_error_f)
            d_total_err += torch.mean(degrees_error_r)
            f_total_err += torch.mean(degrees_error_u)
            
            total += 1.0
            

            if args.write_vector:
                for k in range(len(names)):
                    basename = os.path.basename(names[k]).split(".")[0] + ".txt"
                    with open("./BIWI_results/"+basename, 'w') as f:
                         f.write(str(float(pred_vector_f[k][0]))+" "+str(float(pred_vector_f[k][1]))+" "+str(float(pred_vector_f[k][2]))+'\n')
                         f.write(str(float(pred_vector_r[k][0]))+" "+str(float(pred_vector_r[k][1]))+" "+str(float(pred_vector_r[k][2]))+'\n')
                         f.write(str(float(pred_vector_u[k][0]))+" "+str(float(pred_vector_u[k][1]))+" "+str(float(pred_vector_u[k][2])))

            if args.write_error:
                for k in range(len(names)):
                    if degrees_error[k] > 10.0:
                           error_5['img_name'].append(names[k])
                           error_5['degree_error'].append(float(degrees_error[k]))

            # save euler angle and degrees error to loss_dict
            if args.analysis:
                for k in range(len(names)):
                    loss_dict['img_name'].append(names[k])
                    #loss_dict['angles'].append(angle_label[k].tolist())  # pitch,yaw,roll
                    loss_dict['degree_error_f'].append(float(degrees_error_f[k]))
                    loss_dict['degree_error_r'].append(float(degrees_error_r[k]))
                    loss_dict['degree_error_u'].append(float(degrees_error_u[k]))     
 

            # Save first image in batch with pose cube or axis.
            if args.show_front:
                utils.mkdir(os.path.join(args.save_dir, 'show_front'))
                for j in range(vector_label.size(0)):
                    draw_attention_vector(vector_label[j].cpu().tolist(),
                                          pred_vector[j].cpu().tolist(),
                                          names[j],
                                          args)

    #with open('./loss_pickles/loss_%s.pickle' % (os.path.basename(args.snapshot).split(",")[0]), 'wb') as handle:
    #    pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #print("done saving loss dict.")
    print("Mean degree error for each vector:")
    print("Left Vector:",l_total_err.item() / total)
    print("Down Vector:",d_total_err.item() / total)
    print("Front Vector:",f_total_err.item() / total)

    if args.write_error:
        print("Writing error to local txt file.")
        with open("front_error_10.txt",'w') as f:
                for i in range(len(error_5["img_name"])):
                         f.write(error_5["img_name"][i]+","+str(error_5["degree_error"][i])+'\n')
        print("Done writing.")

    # save analysis of loss distribute
    if args.analysis:
        print('analysis result')
        #utils.show_loss_distribute(loss_dict, os.path.join(args.save_dir, 'analysis'), os.path.basename(args.snapshot).split('.')[0])

    # save collect score curve
    if args.collect_score:
        print("analysis collect score")
        utils.collect_score(loss_dict, os.path.join(args.save_dir, "collect_score"))


if __name__ == '__main__':
    args = parse_args()

    utils.mkdir(args.save_dir)

    # cls and sord
    print("Creating model......")
    if args.model_name == "mobilenetv2":
       model = MobileNetV2(num_classes=args.num_classes)
    else: 
       model = ResNet(torchvision.models.resnet50(pretrained=False),args.num_classes)

    print("Loading weight......")
    saved_state_dict = torch.load(args.snapshot)
    model.load_state_dict(saved_state_dict)
    model.cuda(0)

    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    softmax = nn.Softmax(dim=1).cuda(0)

    # test dataLoader
    test_loader = loadData(args.test_data, args.input_size, args.batch_size, args.num_classes, False)

    # testing
    print('Start testing......')

    if args.collect_score:
        utils.mkdir(os.path.join(args.save_dir, "collect_score"))
    test(model, test_loader, softmax, args)
