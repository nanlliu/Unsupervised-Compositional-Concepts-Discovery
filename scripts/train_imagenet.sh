#!/bin/bash

model_path="stabilityai/stable-diffusion-2-1-base"
train_data_dir="imagenet,imagenet,imagenet,imagenet,imagenet"
placeholder_tokens="<t1>,<t2>,<t3>,<t4>,<t5>"
class_folder_names="n09288635,n02085620,n02481823,n04204347,n03788195"
learnable_property="object,object,object,object,object"
output_dir="output/5_class_compositional_score_s1"

model_path="stabilityai/stable-diffusion-2-1-base"
train_data_dir="imagenet,imagenet,imagenet,imagenet,imagenet"
placeholder_tokens="<t1>,<t2>,<t3>,<t4>,<t5>"
class_folder_names="n02364673,n04552348,n02980441,n02437616,n09472597"
output_dir="output/5_class_compositional_score_s2"
learnable_property="object,object,object,object,object"

model_path="stabilityai/stable-diffusion-2-1-base"
train_data_dir="imagenet,imagenet,imagenet,imagenet,imagenet"
placeholder_tokens="<t1>,<t2>,<t3>,<t4>,<t5>"
class_folder_names="n03100240,n02317335,n04344873,n02504458,n04399382"
output_dir="output/5_class_compositional_score_s3"
learnable_property="object,object,object,object,object"

model_path="stabilityai/stable-diffusion-2-1-base"
train_data_dir="../improved_composable_diffusion/imagenet,../improved_composable_diffusion/imagenet,../improved_composable_diffusion/imagenet,../improved_composable_diffusion/imagenet,../improved_composable_diffusion/imagenet"
placeholder_tokens="<koala>,<ice_bear>,<zebra>,<tiger>,<giant_panda>"
class_folder_names="n01882714,n02134084,n02391049,n02129604,n02510455"
learnable_property="object,object,object,object,object"
output_dir="output/5_class_compositional_score_s4"


DEVICE=$CUDA_VISIBLE_DEVICES
python create_accelerate_config.py --gpu_id "${DEVICE}"
accelerate launch --config_file accelerate_config.yaml main.py \
--pretrained_model_name_or_path "${model_path}" \
--train_data_dir ${train_data_dir}  \
--placeholder_tokens ${placeholder_tokens} \
--resolution=512  --class_folder_names ${class_folder_names} \
--train_batch_size=2 --gradient_accumulation_steps=8 --repeats 1 \
--learning_rate=5.0e-03 --scale_lr --lr_scheduler="constant" --max_train_steps 3000 \
--lr_warmup_steps=0   --output_dir ${output_dir} \
--learnable_property "${learnable_property}"  \
--checkpointing_steps 1000 --mse_coeff 1 --seed 4 \
--add_weight_per_score \
--use_conj_score --init_weight 5 \
--validation_step 1000 \
--num_iters_per_image 120 --num_images_per_class 5