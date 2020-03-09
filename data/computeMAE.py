'''
This file provides modules to do the conversion between Euler angles and Vectors on AFW & AFLW & BIWI datasets

It also include some util functions for visualization

Author:         Zhiwen Cao
Version:        V1.0
Last Update:    Mar-05-2020
'''

import numpy as np
from scipy.optimize import minimize
import math
import re
import os
import time
import cv2
import shutil
import pickle

def draw_vector(img, x, y, tdx=None, tdy=None, size=100, color='r'):
    """
    draw face orientation vector in image
    :param img: face image
    :param x: x of face orientation vector,integer
    :param y: y of face orientation vector,integer
    :param tdx: x of start point,integer
    :param tdy: y of start point,integer
    :param size: length of face orientation vector
    :param color:
    :return:
    """
    if color == 'r':
        color = (0, 0, 255)
    if color == 'g':
        color = (0, 255, 0)
    if color == 'b':
        color = (255, 0, 0)
        
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    x2 = tdx + size * x
    y2 = tdy + size * y

    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2), int(y2)), color, 2, tipLength=0.3)


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    # print('n: ' + str(n))
    return n < 1e-6
  

class AFLW():

    def AFLW_EulerAngles2RotationMatrix(self, rx, ry, rz):
        '''
        rx: pitch
        ry: yaw
        rz: roll
        '''
        ry *= -1
        rz *= -1
        R_x = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(rx), -np.sin(rx)],
                        [0.0, np.sin(rx), np.cos(rx)]])

        R_y = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(ry), 0.0, np.cos(ry)]])

        R_z = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                        [np.sin(rz), np.cos(rz), 0.0],
                        [0.0, 0.0, 1.0]])
                        
        R = R_y @ R_x @ R_z
        return R


    def AFLW_EulerAngles2Vectors(self, rx, ry, rz):
        '''
        rx: pitch
        ry: yaw
        rz: roll
        '''
        T = self.AFLW_EulerAngles2RotationMatrix(rx, ry, rz)
        l_vec = T @ np.array([1, 0, 0])
        b_vec = T @ np.array([0, 1, 0])
        f_vec = T @ np.array([0, 0, 1])
        return l_vec, b_vec, f_vec

    
    # Calculates rotation matrix to euler angles in radians
    def AFLW_RotationMatrixToEulerAngles(self, R) :
        assert(isRotationMatrix(R))
        pitch = -math.asin(R[1, 2])
        roll = -math.atan2(R[1, 0], R[1, 1])
        yaw = -math.atan2(R[0, 2], R[2, 2])

        return np.array([roll, pitch, yaw])


    def AFLW_GetInitGuess(self, l_vec, b_vec, f_vec):
        
        f_vec = np.cross(b_vec, l_vec)
        l_vec = np.cross(f_vec, b_vec)
        
        l_norm = np.linalg.norm(l_vec)
        l_vec /= l_norm
        b_norm = np.linalg.norm(b_vec)
        b_vec /= b_norm
        f_norm = np.linalg.norm(f_vec)
        f_vec /= f_norm

        l_vec = l_vec.reshape(3, 1)
        b_vec = b_vec.reshape(3, 1)
        f_vec = f_vec.reshape(3, 1)
        
        l = np.array([1, 0, 0]).reshape(1, 3)
        b = np.array([0, 1, 0]).reshape(1, 3)
        f = np.array([0, 0, 1]).reshape(1, 3)
        
        R = l_vec @ l + b_vec @ b + f_vec @ f
        
        assert (R.shape == (3, 3))
        
        roll, pitch, yaw = AFLW_RotationMatrixToEulerAngles(R)
        
        return np.array([roll, pitch, yaw])


    def AFLW_Objective(self, x, l_vec, b_vec, f_vec):
        rx = x[0]
        ry = x[1]
        rz = x[2]
        
        l_hat, b_hat, f_hat = AFLW_EulerAngles2Vectors(rx, ry, rz)
        
        return np.sum( (l_hat - l_vec) ** 2 + (b_hat - b_vec) ** 2 + (f_hat - f_vec) **2 )


    def AFLW_ObjectiveV2(self, x, l_vec, b_vec, f_vec):
        rx = x[0]
        ry = x[1]
        rz = x[2]

        l_hat, b_hat, f_hat = AFLW_EulerAngles2Vectors(rx, ry, rz)

        l_vec_dot = np.clip(l_hat[0] * l_vec[0] + l_hat[1] * l_vec[1] + l_hat[2] * l_vec[2], -1, 1)
        b_vec_dot = np.clip(b_hat[0] * b_vec[0] + b_hat[1] * b_vec[1] + b_hat[2] * b_vec[2], -1, 1) 
        f_vec_dot = np.clip(f_hat[0] * f_vec[0] + f_hat[1] * f_vec[1] + f_hat[2] * f_vec[2], -1, 1)
        
        return math.acos(l_vec_dot) + math.acos(b_vec_dot) + math.acos(f_vec_dot)


    def AFLW_ObjectiveV3(self, x, l_vec, b_vec, f_vec):
        rx = x[0]
        ry = x[1]
        rz = x[2]

        l_hat, b_hat, f_hat = AFLW_EulerAngles2Vectors(rx, ry, rz)

        l_vec_dot = np.clip(l_hat[0] * l_vec[0] + l_hat[1] * l_vec[1] + l_hat[2] * l_vec[2], -1, 1)
        b_vec_dot = np.clip(b_hat[0] * b_vec[0] + b_hat[1] * b_vec[1] + b_hat[2] * b_vec[2], -1, 1) 
        f_vec_dot = np.clip(f_hat[0] * f_vec[0] + f_hat[1] * f_vec[1] + f_hat[2] * f_vec[2], -1, 1)
        
        return math.acos(l_vec_dot) ** 2 + math.acos(b_vec_dot) ** 2 + math.acos(f_vec_dot) ** 2


    def AFLW_Get_MAE(self, labels_inpath):
        
        labels_inpath = labels_inpath
        gt_label_inpath = './AFLW_test_labels/'
        gt_img_inpath = './AFLW_test_imgs/'
        bimg_outpath = './AFLW2000_bad_imgs/'
        blabel_outpath = './AFLW2000_bad_labels/'
        gimg_outpath = './AFLW2000_good_imgs/'
        glabel_outpath = './AFLW2000_good_labels/'

        err_dict = {'roll': list(), 'pitch': list(), 'yaw': list()}

        c_rad2deg = 180.0 / np.pi
        c_deg2rad = np.pi / 180.0

        files = sorted([f for f in os.listdir(labels_inpath) if os.path.isfile(os.path.join(labels_inpath, f)) and f.endswith('.txt')])

        cnt = 0
        roll_err_sum = 0
        pitch_err_sum = 0
        yaw_err_sum = 0
        l_vec_err_sum = 0
        b_vec_err_sum = 0
        f_vec_err_sum = 0
        
        x_min, y_min, x_max, y_max = None, None, None, None

        for fname in files:
            roll_deg, pitch_deg, yaw_deg = None, None, None
            pitch_err, yaw_err, roll_err = None, None, None
            with open(labels_inpath + fname, 'r') as f:
                line = f.readline()
                l0, l1, l2 = line.split(' ')
                line = f.readline()
                b0, b1, b2 = line.split(' ')
                line = f.readline()
                f0, f1, f2 = line.split(' ')

                l_vec = np.array([float(l0), float(l1), float(l2)])
                b_vec = np.array([float(b0), float(b1), float(b2)])
                f_vec = np.array([float(f0), float(f1), float(f2)])

                x0 = self.AFLW_GetInitGuess(l_vec, b_vec, f_vec)

                sol = minimize(self.AFLW_ObjectiveV3, x0, args=(l_vec, b_vec, f_vec), method='nelder-mead', options={'xatol': 1e-7, 'disp': False})
                pitch_deg, yaw_deg, roll_deg = sol.x * c_rad2deg



                # For Chart:
                # if fname == 'image00035_0.txt':
                #     print('Predicted vectors:')
                #     print(str(l_vec))
                #     print(str(b_vec))
                #     print(str(f_vec))

                #     l_vec_new, b_vec_new, f_vec_new = AFLW_EulerAngles2Vectors(*sol.x)
                #     print('After Optimization:')
                #     print(str(l_vec_new))
                #     print(str(b_vec_new))
                #     print(str(f_vec_new))

            with open(gt_label_inpath + fname, 'r') as f:
                line = f.readline()
                gt_pitch_rad, gt_yaw_rad, gt_roll_rad = np.array(list(map(float, line.split(','))))
                gt_pitch_deg, gt_yaw_deg, gt_roll_deg = gt_pitch_rad * c_rad2deg, gt_yaw_rad * c_rad2deg, gt_roll_rad * c_rad2deg
                gt_l_vec, gt_b_vec, gt_f_vec = self.AFLW_EulerAngles2Vectors(gt_pitch_rad, gt_yaw_rad, gt_roll_rad)
                our_l_vec, our_b_vec, our_f_vec = self.AFLW_EulerAngles2Vectors(pitch_deg * c_deg2rad, yaw_deg * c_deg2rad, roll_deg * c_deg2rad)

                l_vec_err = math.acos(np.sum(gt_l_vec * our_l_vec)) * c_rad2deg
                b_vec_err = math.acos(np.sum(gt_b_vec * our_b_vec)) * c_rad2deg
                f_vec_err = math.acos(np.sum(gt_f_vec * our_f_vec)) * c_rad2deg

                # gt_roll_rad, gt_pitch_rad, gt_yaw_rad = AFLW_GetInitGuess(gt_l_vec, gt_b_vec, gt_f_vec)
                # gt_pitch_deg, gt_yaw_deg, gt_roll_deg = gt_pitch_rad * c_rad2deg, gt_yaw_rad * c_rad2deg, gt_roll_rad * c_rad2deg

                pitch_err = min( abs(gt_pitch_deg - pitch_deg), abs(pitch_deg + 360 - gt_pitch_deg), abs(pitch_deg - 360 - gt_pitch_deg))
                yaw_err = min( abs(gt_yaw_deg - yaw_deg), abs(yaw_deg + 360 - gt_yaw_deg), abs(yaw_deg - 360 - gt_yaw_deg))
                roll_err = min( abs(gt_roll_deg - roll_deg), abs(roll_deg + 360 - gt_roll_deg), abs(roll_deg - 360 - gt_roll_deg))
                cnt += 1

                #--------------------------- Save to the pickle --------------------------------#
                # err_dict['roll'].append(roll_err)
                # err_dict['pitch'].append(pitch_err)
                # err_dict['yaw'].append(yaw_err)
                #--------------------------------------------------------------------------------#

                # compute accumulated errors
                pitch_err_sum += pitch_err
                yaw_err_sum += yaw_err
                roll_err_sum += roll_err
                l_vec_err_sum += l_vec_err
                b_vec_err_sum += b_vec_err
                f_vec_err_sum += f_vec_err

                #---------------------------- Write Good Images ---------------------------------#
                # if pitch_err < 5 or yaw_err < 5 or roll_err < 5:
                #     img_name = fname.split('.')[0] + '.jpg'
                #     line = f.readline()
                #     x_min, y_min, x_max, y_max = np.array(list(map(int, line.split(','))))
                #     img = cv2.imread(gt_img_inpath + img_name)
                #     img = img[y_min: y_max, x_min: x_max, :]
                    
                #     draw_vector(img, our_l_vec[0], our_l_vec[1], color='b')
                #     draw_vector(img, our_b_vec[0], our_b_vec[1], color='g')
                #     draw_vector(img, our_f_vec[0], our_f_vec[1], color='r')
                #     cv2.imwrite(gimg_outpath + img_name, img)
                #     shutil.copyfile(gt_label_inpath + fname, glabel_outpath + fname)
                #--------------------------------------------------------------------------------#

        
        #--------------------------- Save to the pickle --------------------------------#
        # with open('60_2_0.07.pickle', 'wb') as handle:
        #     pickle.dump(err_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #--------------------------------------------------------------------------------#

        print('VECTORS ERROR:')
        print(l_vec_err_sum / cnt)
        print(b_vec_err_sum / cnt)
        print(f_vec_err_sum / cnt)
        print((l_vec_err_sum + b_vec_err_sum + f_vec_err_sum) / (3 * cnt))
        print('-'*10)
        print('EULER ANGLES ERROR')
        print(roll_err_sum / cnt)
        print(pitch_err_sum / cnt)
        print(yaw_err_sum / cnt)
        print((roll_err_sum + pitch_err_sum + yaw_err_sum) / (3 * cnt))
        
        print('='*10)


###############################################################################
# AFW

class AFW():
    def AFW_GetInitialGuess(self, l_vec, b_vec, f_vec):
        f_vec = np.cross(b_vec, l_vec) * -1
        l_vec = np.cross(f_vec, b_vec) * -1
        
        l_norm = np.linalg.norm(l_vec)
        l_vec /= l_norm
        b_norm = np.linalg.norm(b_vec)
        b_vec /= b_norm
        f_norm = np.linalg.norm(f_vec)
        f_vec /= f_norm
        
        l_vec = l_vec.reshape(3, 1)
        b_vec = b_vec.reshape(3, 1)
        f_vec = f_vec.reshape(3, 1)
            
        l = np.array([1, 0, 0]).reshape(1, 3)
        b = np.array([0, 1, 0]).reshape(1, 3)
        f = np.array([0, 0, 1]).reshape(1, 3)
        
        R = l_vec @ l + b_vec @ b + f_vec @ f
        yaw = math.asin(R[0, 2])
        roll = math.atan2(-R[0, 1], R[0, 0])
        pitch = math.atan2(-R[1, 2], R[2, 2])
        roll *= -1
        return np.array([pitch, yaw, roll])


    def AFW_EulerAngles2Vectors(self, rx, ry, rz):
        rz *= -1
        R_x = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(rx), -np.sin(rx)],
                        [0.0, np.sin(rx), np.cos(rx)]])

        R_y = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(ry), 0.0, np.cos(ry)]])

        R_z = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                        [np.sin(rz), np.cos(rz), 0.0],
                        [0.0, 0.0, 1.0]])
                        
        R = R_x @ R_y @ R_z
        
        l_vec = R @ np.array([1, 0, 0]).T
        b_vec = R @ np.array([0, 1, 0]).T
        f_vec = R @ np.array([0, 0, 1]).T
        return l_vec, b_vec, f_vec


    def AFW_ObjectiveV3(self, x, l_vec, b_vec, f_vec):
        rx = x[0]
        ry = x[1]
        rz = x[2]

        l_hat, b_hat, f_hat = self.AFW_EulerAngles2Vectors(rx, ry, rz)

        l_vec_dot = np.clip(l_hat[0] * l_vec[0] + l_hat[1] * l_vec[1] + l_hat[2] * l_vec[2], -1, 1)
        b_vec_dot = np.clip(b_hat[0] * b_vec[0] + b_hat[1] * b_vec[1] + b_hat[2] * b_vec[2], -1, 1) 
        f_vec_dot = np.clip(f_hat[0] * f_vec[0] + f_hat[1] * f_vec[1] + f_hat[2] * f_vec[2], -1, 1)
        
        return math.acos(l_vec_dot) ** 2 + math.acos(b_vec_dot) ** 2 + math.acos(f_vec_dot) ** 2


    def get_AFW_MAE(self, labels_inpath):
    
        labels_inpath = labels_inpath
        gt_label_inpath = './AFW_Test_Labels/'

        c_rad2deg = 180.0 / np.pi
        c_deg2rad = np.pi / 180.0

        files = sorted([f for f in os.listdir(labels_inpath) if os.path.isfile(os.path.join(labels_inpath, f)) and f.endswith('.txt')])

        cnt = 0
        bad_case_num = 0
        good_case_num = 0
        yaw_err_sum = 0
        for fname in files:
            roll_deg, pitch_deg, yaw_deg = None, None, None
            pitch_err, yaw_err, roll_err = None, None, None
            l_vec, b_vec, f_vec = None, None, None
            with open(labels_inpath + fname, 'r') as f:
                line = f.readline()
                l0, l1, l2 = line.split(' ')
                line = f.readline()
                b0, b1, b2 = line.split(' ')
                line = f.readline()
                f0, f1, f2 = line.split(' ')

                l_vec = np.array([float(l0), float(l1), float(l2)])
                b_vec = np.array([float(b0), float(b1), float(b2)])
                f_vec = np.array([float(f0), float(f1), float(f2)])

                x0 = AFW_GetInitialGuess(l_vec, b_vec, f_vec)
                sol = minimize(AFW_ObjectiveV3, x0, args=(l_vec, b_vec, f_vec), method='nelder-mead', options={'xatol': 1e-7, 'disp': False})
                pitch_deg, yaw_deg, roll_deg = sol.x * c_rad2deg

            with open(gt_label_inpath + fname, 'r') as f:
                line = f.readline()
                gt_pitch_deg, gt_yaw_deg, gt_roll_deg = np.array(list(map(float, line.split(','))))

                yaw_err = min( abs(gt_yaw_deg - yaw_deg), abs(yaw_deg + 360 - gt_yaw_deg), abs(yaw_deg - 360 - gt_yaw_deg))

                line = f.readline()
                line = f.readline()
                gt_l_vec = np.array(list(map(float, line.split(','))))
                line = f.readline()
                gt_b_vec = np.array(list(map(float, line.split(','))))
                line = f.readline()
                gt_f_vec = np.array(list(map(float, line.split(','))))
                
                if yaw_err > 23:
                    #print(yaw_err)
                    bad_case_num += 1

                    print(fname)
                    print('predicted yaw: ' + str(yaw_deg))
                    print('Groud Truth yaw: ' + str(gt_yaw_deg))
                    print('predicted left vector: ' + str(l_vec))
                    print('predicted bottom vector: ' + str(b_vec))
                    print('predicted front vector: ' + str(f_vec))
                    print('Ground truth left vector: ' + str(gt_l_vec))
                    print('Ground truth bottom vector: ' + str(gt_b_vec))
                    print('Ground truth front vector: ' + str(gt_f_vec))
                
                if yaw_err < 8:
                    good_case_num += 1

                yaw_err_sum += yaw_err
                cnt += 1

        print('YAW ANGLES ERROR')
        print(1 - bad_case_num / cnt)
        print('YAW ANGLES GOOD')
        print(good_case_num / cnt)
        # print(yaw_err_sum / cnt)
        print('='*10)

########################################################################3
# 300W

class W300():
    def W300_GetInitialGuess(self, l_vec, b_vec, f_vec):
        f_vec = np.cross(b_vec, l_vec) * -1
        l_vec = np.cross(f_vec, b_vec) * -1
        
        l_norm = np.linalg.norm(l_vec)
        l_vec /= l_norm
        b_norm = np.linalg.norm(b_vec)
        b_vec /= b_norm
        f_norm = np.linalg.norm(f_vec)
        f_vec /= f_norm
        
        l_vec = l_vec.reshape(3, 1)
        b_vec = b_vec.reshape(3, 1)
        f_vec = f_vec.reshape(3, 1)
            
        l = np.array([1, 0, 0]).reshape(1, 3)
        b = np.array([0, 1, 0]).reshape(1, 3)
        f = np.array([0, 0, 1]).reshape(1, 3)
        
        R = l_vec @ l + b_vec @ b + f_vec @ f
        
        yaw = math.asin(R[0, 2])
        roll = math.atan2(-R[0, 1], R[0, 0])
        pitch = math.atan2(-R[1, 2], R[2, 2])
        yaw *= -1
        return np.array([pitch, yaw, roll])


    def W300_EulerAngles2Vectors(self, rx, ry, rz):
        ry *= -1
        R_x = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(rx), -np.sin(rx)],
                        [0.0, np.sin(rx), np.cos(rx)]])

        R_y = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(ry), 0.0, np.cos(ry)]])

        R_z = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                        [np.sin(rz), np.cos(rz), 0.0],
                        [0.0, 0.0, 1.0]])
                        
        R = R_x @ R_y @ R_z
        
        l_vec = R @ np.array([1, 0, 0]).T
        b_vec = R @ np.array([0, 1, 0]).T
        f_vec = R @ np.array([0, 0, 1]).T
        return l_vec, b_vec, f_vec


    def W300_ObjectiveV3(self, x, l_vec, b_vec, f_vec):
        rx = x[0]
        ry = x[1]
        rz = x[2]

        l_hat, b_hat, f_hat = self.W300_EulerAngles2Vectors(rx, ry, rz)

        l_vec_dot = np.clip(l_hat[0] * l_vec[0] + l_hat[1] * l_vec[1] + l_hat[2] * l_vec[2], -1, 1)
        b_vec_dot = np.clip(b_hat[0] * b_vec[0] + b_hat[1] * b_vec[1] + b_hat[2] * b_vec[2], -1, 1) 
        f_vec_dot = np.clip(f_hat[0] * f_vec[0] + f_hat[1] * f_vec[1] + f_hat[2] * f_vec[2], -1, 1)
        
        return math.acos(l_vec_dot) ** 2 + math.acos(b_vec_dot) ** 2 + math.acos(f_vec_dot) ** 2



###########################################################################################33
# YiZhou
class YiZhou():
    def YiZhou_Vectors2RotationMatrix_Best(self, l_vec, b_vec, f_vec):

        l_vec = l_vec / np.linalg.norm(l_vec)

        b_vec_ortho_compo = b_vec - np.sum(l_vec * b_vec) * l_vec

        b_vec = b_vec_ortho_compo / np.linalg.norm(b_vec_ortho_compo)

        f_vec = np.cross(l_vec, b_vec)
        f_vec = f_vec / np.linalg.norm(f_vec)

        R = np.zeros([3,3])

        R[:, 0] = l_vec
        R[:, 1] = b_vec
        R[:, 2] = f_vec
        return R


    def YiZhou_Vectors2RotationMatrix_Worst(self, l_vec, b_vec, f_vec):

        f_vec = f_vec / np.linalg.norm(f_vec)

        b_vec_ortho_compo = b_vec - np.sum(f_vec * b_vec) * f_vec

        b_vec = b_vec_ortho_compo / np.linalg.norm(b_vec_ortho_compo)

        l_vec = np.cross(f_vec, b_vec)
        l_vec = l_vec / np.linalg.norm(l_vec)

        R = np.zeros([3,3])

        R[:, 0] = l_vec
        R[:, 1] = b_vec
        R[:, 2] = f_vec
        return R



    def get_YiZhou_MAE(self):
        
        labels_inpath = './60_2_0/'
        gt_label_inpath = './AFLW_test_labels/'
        gt_img_inpath = './AFLW_test_imgs/'
        bimg_outpath = './AFLW2000_bad_imgs/'
        blabel_outpath = './AFLW2000_bad_labels/'

        c_rad2deg = 180.0 / np.pi
        c_deg2rad = np.pi / 180.0

        files = sorted([f for f in os.listdir(labels_inpath) if os.path.isfile(os.path.join(labels_inpath, f)) and f.endswith('.txt')])

        cnt = 0
        roll_err_sum = 0
        pitch_err_sum = 0
        yaw_err_sum = 0
        l_vec_err_sum = 0
        b_vec_err_sum = 0
        f_vec_err_sum = 0

        for fname in files:
            roll_deg, pitch_deg, yaw_deg = None, None, None
            pitch_err, yaw_err, roll_err = None, None, None
            with open(labels_inpath + fname, 'r') as f:
                line = f.readline()
                l0, l1, l2 = line.split(' ')
                line = f.readline()
                b0, b1, b2 = line.split(' ')
                line = f.readline()
                f0, f1, f2 = line.split(' ')

                l_vec = np.array([float(l0), float(l1), float(l2)])
                b_vec = np.array([float(b0), float(b1), float(b2)])
                f_vec = np.array([float(f0), float(f1), float(f2)])

                R = YiZhou_Vectors2RotationMatrix_Worst(l_vec, b_vec, f_vec)

                aflw = AFLW()

                roll_deg, pitch_deg, yaw_deg = aflw.AFLW_RotationMatrixToEulerAngles(R) * c_rad2deg

            with open(gt_label_inpath + fname, 'r') as f:
                line = f.readline()
                gt_pitch_rad, gt_yaw_rad, gt_roll_rad = np.array(list(map(float, line.split(','))))
                gt_pitch_deg, gt_yaw_deg, gt_roll_deg = gt_pitch_rad * c_rad2deg, gt_yaw_rad * c_rad2deg, gt_roll_rad * c_rad2deg

                pitch_err = min( abs(gt_pitch_deg - pitch_deg), abs(pitch_deg + 360 - gt_pitch_deg), abs(pitch_deg - 360 - gt_pitch_deg))
                yaw_err = min( abs(gt_yaw_deg - yaw_deg), abs(yaw_deg + 360 - gt_yaw_deg), abs(yaw_deg - 360 - gt_yaw_deg))
                roll_err = min( abs(gt_roll_deg - roll_deg), abs(roll_deg + 360 - gt_roll_deg), abs(roll_deg - 360 - gt_roll_deg))
                cnt += 1

                # print('=====================================================')
                # print(cnt)
                # print(roll_deg, pitch_deg, yaw_deg)
                # print(gt_roll_deg, gt_pitch_deg, gt_yaw_deg)

                pitch_err_sum += pitch_err
                yaw_err_sum += yaw_err
                roll_err_sum += roll_err
                # l_vec_err_sum += l_vec_err
                # b_vec_err_sum += b_vec_err
                # f_vec_err_sum += f_vec_err

                # Write Bad cases
                # if pitch_err > 10 or yaw_err > 10 or roll_err > 10:
                #     img_name = fname.split('.')[0] + '.jpg'
                #     img = cv2.imread(gt_img_inpath + img_name)
                #     cv2.imwrite(bimg_outpath + img_name, img)
                #     shutil.copyfile(gt_label_inpath + fname, blabel_outpath + fname)

        print('='*10)
        print('EULER ANGLES ERROR')
        print(roll_err_sum / cnt)
        print(pitch_err_sum / cnt)
        print(yaw_err_sum / cnt)
        print((roll_err_sum + pitch_err_sum + yaw_err_sum) / (3 * cnt))
        print('='*10)
        print('VECTORS ERROR:')
        print(l_vec_err_sum / cnt)
        print(b_vec_err_sum / cnt)
        print(f_vec_err_sum / cnt)
        print((l_vec_err_sum + b_vec_err_sum + f_vec_err_sum) / (3 * cnt))



#-------------------------------------------------- BIWI ----------------------------------------------#
class BIWI():

    def __init__(self, labels_inpath):
        self.labels_inpath = labels_inpath
        self.gt_label_inpath = './BIWI_Test_Labels/'
        # self.gt_img_inpath = ''
        # self.bimg_outpath = './AFLW2000_bad_imgs/'
        # self.blabel_outpath = './AFLW2000_bad_labels/'
        # self.gimg_outpath = './AFLW2000_good_imgs/'
        # self.glabel_outpath = './AFLW2000_good_labels/'

        self.c_rad2deg = 180.0 / np.pi
        self.c_deg2rad = np.pi / 180.0


    def BIWI_GetInitGuess(self, v1, v2, v3):
        '''
        BIWI : pitch -> yaw -> roll (right hand coordinate system)
        '''
        v3 = np.cross(v1, v2)
        v2 = np.cross(v3, v1)
    
        v1_norm = np.linalg.norm(v1)
        v1 /= v1_norm
        v2_norm = np.linalg.norm(v2)
        v2 /= v2_norm
        v3_norm = np.linalg.norm(v3)
        v3 /= v3_norm
    
        v1 = v1.reshape(3, 1)
        v2 = v2.reshape(3, 1)
        v3 = v3.reshape(3, 1)
            
        l = np.array([1, 0, 0]).reshape(1, 3)
        b = np.array([0, 1, 0]).reshape(1, 3)
        f = np.array([0, 0, 1]).reshape(1, 3)
        
        R = v1 @ l + v2 @ b + v3 @ f
        assert isRotationMatrix(R)

        yaw = -math.asin(R[2, 0])
        pitch = math.atan2(R[2, 1], R[2, 2])
        roll = math.atan2(R[1, 0], R[0, 0])

        return np.array([pitch, yaw, roll]) # in radians


    def BIWI_EulerAngles2Vectors(self, rx, ry, rz):
        R_x = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(rx), -np.sin(rx)],
                        [0.0, np.sin(rx), np.cos(rx)]])

        R_y = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(ry), 0.0, np.cos(ry)]])

        R_z = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                        [np.sin(rz), np.cos(rz), 0.0],
                        [0.0, 0.0, 1.0]])
                    
        R = R_z @ R_y @ R_x
        v0 = R[:, 0]
        v1 = R[:, 1]
        v2 = R[:, 2]
        return v0, v1, v2

    def BIWI_ObjectiveV2(self, x, l_vec, b_vec, f_vec):
        rx = x[0]
        ry = x[1]
        rz = x[2]
        
        l_hat, b_hat, f_hat = self.BIWI_EulerAngles2Vectors(rx, ry, rz)
        
        return np.sum( (l_hat - l_vec) ** 2 + (b_hat - b_vec) ** 2 + (f_hat - f_vec) **2 )

    def BIWI_ObjectiveV3(self, x, l_vec, b_vec, f_vec):
        rx = x[0]
        ry = x[1]
        rz = x[2]

        l_hat, b_hat, f_hat = self.BIWI_EulerAngles2Vectors(rx, ry, rz)

        l_vec_dot = np.clip(l_hat[0] * l_vec[0] + l_hat[1] * l_vec[1] + l_hat[2] * l_vec[2], -1, 1)
        b_vec_dot = np.clip(b_hat[0] * b_vec[0] + b_hat[1] * b_vec[1] + b_hat[2] * b_vec[2], -1, 1) 
        f_vec_dot = np.clip(f_hat[0] * f_vec[0] + f_hat[1] * f_vec[1] + f_hat[2] * f_vec[2], -1, 1)
        
        return math.acos(l_vec_dot) ** 2 + math.acos(b_vec_dot) ** 2 + math.acos(f_vec_dot) ** 2


    def BIWI_GetMAE(self):
        cnt = 0
        roll_err_sum = 0
        pitch_err_sum = 0
        yaw_err_sum = 0
        l_vec_err_sum = 0
        b_vec_err_sum = 0
        f_vec_err_sum = 0

        label_names = sorted([f for f in os.listdir(self.labels_inpath) if os.path.isfile(os.path.join(self.labels_inpath, f)) and f.endswith('.txt')])

        for label_name in label_names:
            roll_deg, pitch_deg, yaw_deg = None, None, None
            pitch_err, yaw_err, roll_err = None, None, None
            with open(self.labels_inpath + label_name, 'r') as f:
                line = f.readline()
                l0, l1, l2 = line.split(' ')
                line = f.readline()
                b0, b1, b2 = line.split(' ')
                line = f.readline()
                f0, f1, f2 = line.split(' ')

                l_vec = np.array([float(l0), float(l1), float(l2)])
                b_vec = np.array([float(b0), float(b1), float(b2)])
                f_vec = np.array([float(f0), float(f1), float(f2)])

                x0 = self.BIWI_GetInitGuess(l_vec, b_vec, f_vec)

                sol = minimize(self.BIWI_ObjectiveV3, x0, args=(l_vec, b_vec, f_vec), method='nelder-mead', options={'xatol': 1e-7, 'disp': False})
                pitch_deg, yaw_deg, roll_deg = sol.x * self.c_rad2deg

            with open(self.gt_label_inpath + label_name, 'r') as f:
                line = f.readline()
                gt_pitch_deg, gt_yaw_deg, gt_roll_deg = np.array(list(map(float, line.split(','))))
                
                gt_pitch_rad, gt_yaw_rad, gt_roll_rad = gt_pitch_deg * self.c_deg2rad, gt_yaw_deg * self.c_deg2rad, gt_roll_deg * self.c_deg2rad
                gt_l_vec, gt_b_vec, gt_f_vec = self.BIWI_EulerAngles2Vectors(gt_pitch_rad, gt_yaw_rad, gt_roll_rad)
                our_l_vec, our_b_vec, our_f_vec = self.BIWI_EulerAngles2Vectors(pitch_deg * self.c_deg2rad, yaw_deg * self.c_deg2rad, roll_deg * self.c_deg2rad)

                l_vec_err = math.acos(np.sum(gt_l_vec * our_l_vec)) * self.c_rad2deg
                b_vec_err = math.acos(np.sum(gt_b_vec * our_b_vec)) * self.c_rad2deg
                f_vec_err = math.acos(np.sum(gt_f_vec * our_f_vec)) * self.c_rad2deg

                pitch_err = min( abs(gt_pitch_deg - pitch_deg), abs(pitch_deg + 360 - gt_pitch_deg), abs(pitch_deg - 360 - gt_pitch_deg))
                yaw_err = min( abs(gt_yaw_deg - yaw_deg), abs(yaw_deg + 360 - gt_yaw_deg), abs(yaw_deg - 360 - gt_yaw_deg))
                roll_err = min( abs(gt_roll_deg - roll_deg), abs(roll_deg + 360 - gt_roll_deg), abs(roll_deg - 360 - gt_roll_deg))

                if label_name == '03_frame_00714_pose.txt':
                    print(label_name)
                    print('pitch: ' + str(pitch_err))
                    print('our vec_l: ' + str(our_l_vec))
                    print('our vec_b: ' + str(our_b_vec))
                    print('our vec_f: ' + str(our_f_vec))
                    print('true roll: ' + str(gt_roll_deg))
                    print('true pitch: ' + str(gt_pitch_deg))
                    print('true yaw: ' + str(gt_yaw_deg))
                    print('our roll: ' + str(roll_deg))
                    print('our pitch: ' + str(pitch_deg))
                    print('our yaw: ' + str(yaw_deg))


                # if pitch_err > 60: 
                #     print(label_name)
                #     print('pitch: ' + str(pitch_err))
                #     print('our vec_l: ' + str(our_l_vec))
                #     print('our vec_b: ' + str(our_b_vec))
                #     print('our vec_f: ' + str(our_f_vec))
                #     print('true roll: ' + str(gt_roll_deg))
                #     print('true pitch: ' + str(gt_pitch_deg))
                #     print('true yaw: ' + str(gt_yaw_deg))
                #     print('our roll: ' + str(roll_deg))
                #     print('our pitch: ' + str(pitch_deg))
                #     print('our yaw: ' + str(yaw_deg))
                    
                # if yaw_err > 60:
                #     print(label_name)
                #     print('yaw: ' + str(yaw_err))
                #     print('our vec_l: ' + str(our_l_vec))
                #     print('our vec_b: ' + str(our_b_vec))
                #     print('our vec_f: ' + str(our_f_vec))
                #     print('true roll: ' + str(gt_roll_deg))
                #     print('true pitch: ' + str(gt_pitch_deg))
                #     print('true yaw: ' + str(gt_yaw_deg))
                #     print('our roll: ' + str(roll_deg))
                #     print('our pitch: ' + str(pitch_deg))
                #     print('our yaw: ' + str(yaw_deg))
                    
                # if roll_err > 60:
                #     print(label_name)
                #     print('roll: ' + str(roll_err))
                #     print('our vec_l: ' + str(our_l_vec))
                #     print('our vec_b: ' + str(our_b_vec))
                #     print('our vec_f: ' + str(our_f_vec))
                #     print('true roll: ' + str(gt_roll_deg))
                #     print('true pitch: ' + str(gt_pitch_deg))
                #     print('true yaw: ' + str(gt_yaw_deg))
                #     print('our roll: ' + str(roll_deg))
                #     print('our pitch: ' + str(pitch_deg))
                #     print('our yaw: ' + str(yaw_deg))
                    
                cnt += 1

                # print(pitch_err, yaw_err, roll_err)

                # compute accumulated errors
                pitch_err_sum += pitch_err
                yaw_err_sum += yaw_err
                roll_err_sum += roll_err
                l_vec_err_sum += l_vec_err
                b_vec_err_sum += b_vec_err
                f_vec_err_sum += f_vec_err
        
        print('VECTORS ERROR:')
        print(l_vec_err_sum / cnt)
        print(b_vec_err_sum / cnt)
        print(f_vec_err_sum / cnt)
        print((l_vec_err_sum + b_vec_err_sum + f_vec_err_sum) / (3 * cnt))
        print('-'*10)
        print('EULER ANGLES ERROR')
        print(roll_err_sum / cnt)
        print(pitch_err_sum / cnt)
        print(yaw_err_sum / cnt)
        print((roll_err_sum + pitch_err_sum + yaw_err_sum) / (3 * cnt))
        
        print('='*10)


    def test(self):
        # v1 = np.array([0.9971307848457988,-0.042406591104648665,-0.06270469634472936])
        # v2 = np.array([0.033603233843610227,0.9902312683298323,-0.13532500839513453])
        # v3 = np.array([0.06783082328892398,0.1328296512559403,0.9888150803659762])
        # pitch, yaw, roll = 13,22,-25
        roll, pitch, yaw = 27, 40, 50
        pitch *= self.c_deg2rad
        roll *= self.c_deg2rad
        yaw *= self.c_deg2rad
        nv1, nv2, nv3 = self.BIWI_EulerAngles2Vectors(pitch, yaw, roll)

        # assert all(v1 - nv1 < 1e-6) and all(v2 - nv2 < 1e-6) and all(v3 - nv3 < 1e-6)
        newp, newy, newr = self.BIWI_GetInitGuess(nv1, nv2, nv3)
        print(newr * self.c_rad2deg)
        print(newp * self.c_rad2deg)
        print(newy * self.c_rad2deg)
        # assert newp - pitch < 0.1 and newy - yaw < 0.1 and newr - roll < 0.1
        print('ok')
        

if __name__ == '__main__':
    pass
    #----------------------------------------------- AFW --------------------------------------------------------#

    # bins = [60, 90, 120]
    # alpha = [1, 2]
    # beta = [0.03, 0.05, 0.07]

    # for bin in bins:
    #     for a in alpha:
    #         for b in beta:
    #             path = './MobileNet_AFLW2000/' + str(bin) + '_' + str(a) + '_' + str(b) + '/'
    #             print(path)
    #             get_MAE(path)

    
    # get_AFW_MAE('BIWI_Test_Results/AFW_results/')

    #----------------------------------------------- AFLW --------------------------------------------------------#

    # get_AFLW_MAE('ResNet_AFLW2000_Flip/AFLW_results/')


    # bins = [40, 60]
    # alpha = [1, 2]
    # beta = [0.05, 0.07]
    # for bin in bins:
    #     for a in alpha:
    #         for b in beta:
    #             path = 'ResNet_AFLW2000_Flip/' + str(bin) + '_' + str(a) + '_' + str(b) + '/AFLW_results/'
    #             print(str(bin) + '_' + str(a) + '_' + str(b))
    #             get_AFLW_MAE(path)
    # get_AFLW_MAE('ResNet_AFLW2000_Flip/60_2_0.07/AFLW_results/')
    # get_AFLW_MAE('./ResNet_AFLW2000/60_2_0.07/')

    #----------------------------------------------- BIWI --------------------------------------------------------#
    # biwi = BIWI('./Ablation/withoutCls/BIWI_results/')
    # biwi = BIWI('./BIWI_Test_Results/BIWI_results/')
    # biwi = BIWI('./Ablation/MobileNet/BIWI_results/')
    # biwi.BIWI_GetMAE()
    # biwi.test()