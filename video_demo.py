import cv2
from test_on_img import Test
from utils import remove_distortion
from net import MobileNetV2
from torchvision import transforms
from net import MobileNetV2
import argparse
import matplotlib.pyplot as plt
from net import MobileNetV2, ResNet
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import torchvision


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Video Testing.')
  
    parser.add_argument('--model_name', dest='model_name', help='mode name', default='', type=str)
    parser.add_argument('--video', dest='video', help='Directory path for data.',
                        default='', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='model path',
                        default='', type=str)
    parser.add_argument('--num_classes', dest="num_classes", help="bins", default=66, type=int)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    _, frame = cap.read()

    test = Test(args.model_name, args.snapshot, args.num_classes)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    args = parse_args()

    # MTCNN face detector
    detector = MTCNN()

    # read video stream
    cap = cv2.VideoCapture(args.video)
    _, frame = cap.read()


    # load model
    test = Test(args.model_name, args.snapshot, args.num_classes)
    # image pre-processing
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    frame_cnt = 1
    pre_frame = frame

    while 1:
        _, frame = cap.read()
        draw_img = frame.copy()
    

        if frame_cnt % 3 != 0:
            frame_cnt += 1
            cv2.imshow("pose visualization", pre_frame)
            continue
        
        detect_res = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_cnt += 1

        if detect_res:
            for i in range(len(detect_res)):
                # if confidence is below 0.9, we skip detection
                if detect_res[i]['confidence'] < 0.98:
                    continue
                #crop faces
                x,y,w,h = detect_res[i]['box']
                center = [x + w//2, y + h//2]
                img = frame[y:y+h,x:x+w]
                
                img = cv2.resize(img,(224,224)) #remove comment if Resize() does not work.
                img = transform(img)
                img = img.unsqueeze(0)
            
                test.test_per_img(img,draw_img, center, w)
            pre_frame = draw_img

        else:
            pre_frame = draw_img
            cv2.imshow("pose visualization", pre_frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
 
    cv2.destroyAllWindows()
    cap.release()
	
	
