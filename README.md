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
