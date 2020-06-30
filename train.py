# -*- coding:utf-8 -*-
"""
    training HeadPoseNet
"""
import os
import utils
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from log import Logger
from dataset import loadData
from net import ResNet
from tensorboardX import SummaryWriter
from torchvision.models.mobilenet import model_urls
import torchvision
from loss import FocalLoss

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="TriNet: Head Pose Estimation")
    parser.add_argument("--epochs", dest="epochs", help="Maximum number of training epochs.",
                        default=20, type=int)
    parser.add_argument("--basenet", dest="basenet",help="choice of basenet", nargs="+", default="resnet50")
    parser.add_argument("--alpha_reg", dest="alpha_reg", help="regression coef", nargs="+", default="1.0 2.0")
    parser.add_argument("--beta", dest="beta", help="ortho coef", nargs="+", default="0.3 0.6 0.9")
    parser.add_argument("--num_bins", dest="num_bins", help="number of bins", nargs="+", default="40 60")
    parser.add_argument("--batch_size", dest="batch_size", help="batch size",
                        default=64, type=int)
    parser.add_argument("--lr_resnet", dest="lr_resnet", help="Base learning rate",
                        default=0.00001, type=float)
    parser.add_argument("--lr_mobilenet", dest="lr_mobilenet", help="", default=0.0001, type=float)
    parser.add_argument("--lr_decay", dest="lr_decay", help="learning rate decay rate",
                        default=0.95, type=float)
    parser.add_argument("--save_dir", dest="save_dir", help="directory path of saving results",
                        default='./experiments', type=str)
    parser.add_argument("--train_data", dest="train_data", help="directory path of train dataset",
                        default="", type=str)
    parser.add_argument("--valid_data", dest="valid_data", help="directory path of valid dataset",
                        default="", type=str)
    parser.add_argument("--snapshot", dest="snapshot", help="pre trained weight path",
                        default="", type=str)
    parser.add_argument("--unfreeze", dest="unfreeze", help="unfreeze some layer after several epochs",
                        default=5, type=int)
    parser.add_argument("--width_mult", dest="width_mult", choices=[0.5, 1.0], help="mobile V2 width_mult",
                        default=1.0, type=float)
    parser.add_argument("--input_size", dest="input_size", choices=[224, 192, 160, 128, 96], help="size of input images",
                        default=224, type=int)
    # training loss
    parser.add_argument("--ortho_loss", default=True, type=bool)
    parser.add_argument("--cls_loss", choices=['KLDiv', 'BCE', 'CrossEntropy', 'FocalLoss'], help="loss type for bin classification", default='BCE', type=str)
    parser.add_argument("--reg_loss", choices=['acos','value'], help='regression target', default='acos', type=str)
    args = parser.parse_args()
    return args


def get_non_ignored_params(model, model_name):
    # Generator function that yields params that will be optimized.
    if model_name == 'resnet50':
        b = [model.features]
    else:
        b = [model.features]
    
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_cls_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_x1, model.fc_y1, model.fc_z1, model.fc_x2, model.fc_y2, model.fc_z2, model.fc_x3, model.fc_y3, model.fc_z3]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def valid(model, valid_loader, softmax, num_classes):
    """
    Validation on test images

    return: 
            Mean angle errors between predicted vectors and groud truth vectors
    """
    degrees_error_v1 = 0.0
    degrees_error_v2 = 0.0
    degrees_error_v3 = 0.0
    batch_num = 0.0
    model.eval()
    with torch.no_grad():

        for j, (valid_img, cls_v1, cls_v2, cls_v3, reg_v1, reg_v2, reg_v3, _, _, _, _) in enumerate(valid_loader):
            valid_img = valid_img.cuda(0).float()

            reg_v1 = reg_v1.cuda(0)
            reg_v2 = reg_v2.cuda(0)
            reg_v3 = reg_v3.cuda(0)

            # get x,y,z cls predictions
            x_pred_v1, y_pred_v1, z_pred_v1, x_pred_v2, y_pred_v2, z_pred_v2, x_pred_v3, y_pred_v3, z_pred_v3 = model(valid_img)

            # get prediction vector(get continue value from classify result)
            _, _, _, vector_pred_v1 = utils.classify2vector(x_pred_v1, y_pred_v1, z_pred_v1, softmax, num_classes)
            _, _, _, vector_pred_v2 = utils.classify2vector(x_pred_v2, y_pred_v2, z_pred_v2, softmax, num_classes)
            _, _, _, vector_pred_v3 = utils.classify2vector(x_pred_v3, y_pred_v3, z_pred_v3, softmax, num_classes)

            # get validation degrees error
            cos_value = utils.vector_cos(vector_pred_v1, reg_v1)
            degrees_error_v1 += torch.mean(torch.acos(cos_value) * 180 / np.pi)

            cos_value = utils.vector_cos(vector_pred_v2, reg_v2)
            degrees_error_v2 += torch.mean(torch.acos(cos_value) * 180 / np.pi)

            cos_value = utils.vector_cos(vector_pred_v3, reg_v3)
            degrees_error_v3 += torch.mean(torch.acos(cos_value) * 180 / np.pi)

            batch_num += 1.0

    return degrees_error_v1 / batch_num, degrees_error_v2 / batch_num, degrees_error_v3 / batch_num

def train(net, bins, alpha, beta, batch_size):
    """
    params: 
          bins: number of bins for classification
          alpha: regression loss weight
          beta: ortho loss weight
    """ 
    # create model
    if net == "resnet50":
          model = ResNet(torchvision.models.resnet50(pretrained=True), num_classes=bins)
          lr = args.lr_resnet
    else:
          model = MobileNetV2(bins)
          lr = args.lr_mobilenet

    # loading data
    logger.logger.info("Loading data".center(100, '='))
    train_data_loader = loadData(args.train_data, args.input_size, batch_size, bins)
    valid_data_loader = loadData(args.valid_data, args.input_size, batch_size, bins, False) 

    # initialize cls loss function
    if args.cls_loss == "KLDiv":
        cls_criterion = nn.KLDivLoss(reduction='batchmean').cuda(0)
    elif args.cls_loss == "BCE":
        cls_criterion = nn.BCELoss().cuda(0)
    elif args.cls_loss == 'FocalLoss':
        cls_criterion = FocalLoss(bins).cuda(0)
    elif args.cls_loss == 'CrossEntropy':
        cls_criterion = nn.CrossEntropyLoss().cuda(0)
    
    # initialize reg loss function
    reg_criterion = nn.MSELoss().cuda(0)
    softmax = nn.Softmax(dim=1).cuda(0)
    sigmoid = nn.Sigmoid().cuda(0)
    model.cuda(0)
    
    # training log
    logger.logger.info("Training".center(100, '='))

    # initialize learning rate and step
    lr = lr
    step = 0

    # validation error
    min_avg_error = 1000.

    # start training
    for epoch in range(args.epochs):
        print("Epoch:", epoch)
        model.train()
        # learning rate initialization
        if net == 'resnet50':
            if epoch >= args.unfreeze:
                optimizer = torch.optim.Adam([{"params": get_non_ignored_params(model, net), "lr": lr},
                                          {"params": get_cls_fc_params(model), "lr": lr * 10}], lr=args.lr_resnet)
            else:
                optimizer = torch.optim.Adam([{"params": get_non_ignored_params(model, net), "lr": lr},
                                          {"params": get_cls_fc_params(model), "lr": lr * 10}], lr=args.lr_resnet)

        else:
            if epoch >= args.unfreeze:
                optimizer = torch.optim.Adam([{"params": get_non_ignored_params(model, net), "lr": lr},
                                          {"params": get_cls_fc_params(model), "lr": lr}], lr=args.lr_mobilenet)
            else:
                optimizer = torch.optim.Adam([{"params": get_non_ignored_params(model, net), "lr": lr * 10},
                                          {"params": get_cls_fc_params(model), "lr": lr * 10}], lr=args.lr_mobilenet)

        # reduce lr by lr_decay factor for each epoch
        lr = lr * args.lr_decay
        print("------------")

        for i, (images, cls_v1, cls_v2, cls_v3, reg_v1, reg_v2, reg_v3, name, left_targets, down_targets, front_targets) in enumerate(train_data_loader):
            images = images.cuda(0).float()
            
            # get classified labels
            cls_v1 = cls_v1.cuda(0)
            cls_v2 = cls_v2.cuda(0)
            cls_v3 = cls_v3.cuda(0)

            # get continuous labels
            reg_v1 = reg_v1.cuda(0)
            reg_v2 = reg_v2.cuda(0)
            reg_v3 = reg_v3.cuda(0)

            left_targets = left_targets.cuda(0)
            down_targets = down_targets.cuda(0)
            front_targets = front_targets.cuda(0)

            # inference
            x_pred_v1, y_pred_v1, z_pred_v1, x_pred_v2, y_pred_v2, z_pred_v2, x_pred_v3, y_pred_v3, z_pred_v3 = model(images)

            logits = [x_pred_v1, y_pred_v1, z_pred_v1, x_pred_v2, y_pred_v2, z_pred_v2, x_pred_v3, y_pred_v3, z_pred_v3]

            loss, degree_error_v1, degree_error_v2, degree_error_v3 = utils.computeLoss(cls_v1, cls_v2, cls_v3, reg_v1, reg_v2, reg_v3, 
                                                                    logits, softmax, sigmoid, cls_criterion, reg_criterion, left_targets, down_targets, front_targets, [bins, alpha, beta, args.cls_loss, args.reg_loss, args.ortho_loss])

            # backward
            grad = [torch.tensor(1.0).cuda(0) for _ in range(3)]
            optimizer.zero_grad()
            torch.autograd.backward(loss, grad)
            optimizer.step()

            # save training log and weight
            if (i + 1) % 500 == 0:
                msg = "Epoch: %d/%d | Iter: %d/%d | x_loss: %.6f | y_loss: %.6f | z_loss: %.6f | degree_error_f:%.3f | degree_error_r:%.3f | degree_error_u:%.3f"  % (
                    epoch, args.epochs, i + 1, len(train_data_loader.dataset) // batch_size, loss[0].item(), loss[1].item(),
                    loss[2].item(), degree_error_v1.item(), degree_error_v2.item(), degree_error_v3.item())
                print(msg)
                logger.logger.info(msg)

        # Test on validation dataset
        error_v1, error_v2, error_v3 = valid(model, valid_data_loader, softmax, bins)
        print("Epoch:", epoch)
        print("Validation Error:", error_v1.item(), error_v2.item(), error_v3.item())
        logger.logger.info("Validation Error(l,d,f)_{},{},{}".format(error_v1.item(), error_v2.item(), error_v3.item()))        
            
        # save model if achieve better validation performance
        if error_v1.item() + error_v2.item() + error_v3.item() < min_avg_error:

            min_avg_error = error_v1.item() + error_v2.item() + error_v3.item()
            print("Training Info:")
            print("Model:", net, " ","Number of bins:",bins, " ", "Alpha:", alpha, " ", "Beta:", beta)
            print("Saving Model......")
            torch.save(model.state_dict(), os.path.join(snapshot_dir, output_string + '_Best_' + '.pkl'))
            print("Saved")



if __name__ == "__main__":
    args = parse_args()

    num_bins = list(map(int, args.num_bins.split(" ")))
    alphas = list(map(float, args.alpha_reg.split(" ")))
    betas = list(map(float, args.beta.split(" ")))
    basenets = args.basenet.split(" ")
    
    for net in basenets:
        for bins in num_bins:
            for alpha in alphas:
                for beta in betas:
                    output_string = "ALL_%s_bins_%d_alpha_%f_beta_%f" % (net, bins, alpha, beta)
                    
                    
                    # mkdir
                    project_dir = os.path.join(args.save_dir, output_string)
                    utils.mkdir(project_dir)
                    snapshot_dir = os.path.join(project_dir, "snapshot")
                    utils.mkdir(snapshot_dir)
                    summary_dir = os.path.join(project_dir, "summary")
                    utils.mkdir(summary_dir)
                    log_path = os.path.join(project_dir, "training.log_%d_%s_%f_%f" % (bins, net, alpha, beta))
         

                    # create summary writer and log
                    writer = SummaryWriter(log_dir=summary_dir)
                    logger = Logger(log_path, 'info')


                    # print parameters
                    #logger.logger.info("Parameters".center(100, '='))
                    logger.logger.info("\nmodel:%s\nbins:%d\nalpha:%f\nbeta:%f\n" % (
                        net, bins, alpha, beta))
                    
                    if net == 'mobilenetv2':
                        batch_size = 128
                    else:
                        batch_size = 64

                    # run train function
                    train(net, bins, alpha, beta, batch_size)
