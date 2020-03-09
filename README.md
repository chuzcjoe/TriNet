# TriNet
TriNet for 3DOF object pose estimation
<img src="https://github.com/chuzcjoe/TriNet/raw/master/imgs/demo.jpg" width="800">

## Network Overview
<img src="https://github.com/chuzcjoe/TriNet/raw/master/imgs/NN.png" width="800">

## Train
python train.py --train_data --valid_data --basenet --alpha_reg --beta --num_bins --batch_size --save_dir --ortho_loss --cls_loss --reg_loss

## Test
python test.py --test_data --snapsot --num_classes --model_name 

## Video Test
./video.sh

## Webcam Test
./webcam.sh

## Dependencies:
matplotlib==3.1.3                    
numpy==1.18.1                   
opencv-python==4.1.1.26                 
pandas==1.0.1                    
pillow==7.0.0                    
pyparsing==2.4.6                   
python==3.7.6                
scipy==1.4.1                    
seaborn==0.10.0                   
setuptools==45.2.0                   
sqlite==3.31.1                
tensorboardx==2.0                      
tk==8.6.8                 
torch==1.4.0                    
torchvision==0.5.0                    
tqdm==4.43.0             

## Data Pre-processing

Three datasets are involved: BIWI, AFLW, 300W_LP. We describe how we obtain three directional vectors in data/computeMAE.py file

## Vector Refinement

We refine three network output vectors by solving a optimization problem. See implementation details in Optimization/optimize.py

## Experiments
### Comparison to state-of-the-art methods
| Method  |  roll | pitch  | yaw  |  MAE |
|---|---|---|---|---|
| Dlib  | 10.545  | 13.633  | 23.153  | 15.777  |
|   |   |   |   |   |
|   |   |   |   |   |
|   |   |   |   |   |
|   |   |   |   |   |
|   |   |   |   |   |
|   |   |   |   |   |

### AFLW2000
<img src="https://github.com/chuzcjoe/TriNet/raw/master/imgs/Euler%20Angles%20Errors.png" width="800">

### AFW
<img src="https://github.com/chuzcjoe/TriNet/raw/master/imgs/AFW.png" width="500">

