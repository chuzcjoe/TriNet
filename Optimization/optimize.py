import numpy as np
import math
from scipy.optimize import minimize



class Optimize():
    
    def __init__(self):
        self.c_rad2deg = 180.0 / np.pi
        self.c_deg2rad = np.pi / 180.0


    def isRotationMatrix(self, R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        # print('n: ' + str(n))
        return n < 1e-6


    def Rot_Matrix_2_Euler_Angles(self, R):
        assert(self.isRotationMatrix(R))
        pitch = -math.asin(R[1, 2])
        roll = -math.atan2(R[1, 0], R[1, 1])
        yaw = -math.atan2(R[0, 2], R[2, 2])

        return np.array([roll, pitch, yaw])


    def Get_Init_Guess(self, l_vec, b_vec, f_vec):
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
        
        roll, pitch, yaw = self.Rot_Matrix_2_Euler_Angles(R)
        
        return np.array([roll, pitch, yaw])

    def Euler_Angles_2_Vectors(self, rx, ry, rz):
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
        l_vec = R @ np.array([1, 0, 0])
        b_vec = R @ np.array([0, 1, 0])
        f_vec = R @ np.array([0, 0, 1])
        return np.array([l_vec, b_vec, f_vec])


    def Objective(self, x, l_vec, b_vec, f_vec):
        rx = x[0]
        ry = x[1]
        rz = x[2]

        l_hat, b_hat, f_hat = self.Euler_Angles_2_Vectors(rx, ry, rz)

        l_vec_dot = np.clip(l_hat[0] * l_vec[0] + l_hat[1] * l_vec[1] + l_hat[2] * l_vec[2], -1, 1)
        b_vec_dot = np.clip(b_hat[0] * b_vec[0] + b_hat[1] * b_vec[1] + b_hat[2] * b_vec[2], -1, 1) 
        f_vec_dot = np.clip(f_hat[0] * f_vec[0] + f_hat[1] * f_vec[1] + f_hat[2] * f_vec[2], -1, 1)
        
        return math.acos(l_vec_dot) ** 2 + math.acos(b_vec_dot) ** 2 + math.acos(f_vec_dot) ** 2


    def Get_Ortho_Vectors(self, l_vec, b_vec, f_vec):
        
        x0 = self.Get_Init_Guess(l_vec, b_vec, f_vec)
        sol = minimize(self.Objective, x0, args=(l_vec, b_vec, f_vec), method='nelder-mead', options={'xatol': 1e-7, 'disp': False})
        pitch_rad, yaw_rad, roll_rad = sol.x 

        v1, v2, v3 = self.Euler_Angles_2_Vectors(pitch_rad, yaw_rad, roll_rad)

        return np.array([v1, v2, v3])