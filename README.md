# TriNet
TriNet for 3DOF object pose estimation

## Train
python train.py --train_data --valid_data --basenet --alpha_reg --beta --num_bins --batch_size --save_dir --ortho_loss --cls_loss --reg_loss

## Test
python test.py --test_data --snapsot --num_classes --model_name 

## Video Test
./video.sh

## Webcam Test
./webcam.sh

## Requirements:
ca-certificates           2020.1.1                      0  
certifi                   2019.11.28               py37_0  
cycler                    0.10.0                   pypi_0    pypi
kiwisolver                1.1.0                    pypi_0    pypi
ld_impl_linux-64          2.33.1               h53a641e_7  
libedit                   3.1.20181209         hc058e9b_0  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 9.1.0                hdf63c60_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
matplotlib                3.1.3                    pypi_0    pypi
ncurses                   6.2                  he6710b0_0  
numpy                     1.18.1                   pypi_0    pypi
opencv-python             4.1.1.26                 pypi_0    pypi
openssl                   1.1.1d               h7b6447c_4  
pandas                    1.0.1                    pypi_0    pypi
pillow                    7.0.0                    pypi_0    pypi
pip                       20.0.2                   py37_1  
protobuf                  3.11.3                   pypi_0    pypi
pyparsing                 2.4.6                    pypi_0    pypi
python                    3.7.6                h0371630_2  
python-dateutil           2.8.1                    pypi_0    pypi
pytz                      2019.3                   pypi_0    pypi
readline                  7.0                  h7b6447c_5  
scipy                     1.4.1                    pypi_0    pypi
seaborn                   0.10.0                   pypi_0    pypi
setuptools                45.2.0                   py37_0  
six                       1.14.0                   pypi_0    pypi
sqlite                    3.31.1               h7b6447c_0  
tensorboardx              2.0                      pypi_0    pypi
tk                        8.6.8                hbc83047_0  
torch                     1.4.0                    pypi_0    pypi
torchvision               0.5.0                    pypi_0    pypi
tqdm                      4.43.0                   pypi_0    pypi
wheel                     0.34.2                   py37_0  
xz                        5.2.4                h14c3975_4  
zlib                      1.2.11               h7b6447c_3  
