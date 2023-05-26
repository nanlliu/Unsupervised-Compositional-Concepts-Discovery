#!/bin/bash

model_path="stabilityai/stable-diffusion-2-1-base"
train_data_dir="ADE20K/home_or_hotel"
placeholder_tokens="<t1>,<t2>,<t3>,<t4>,<t5>"
class_folder_names="kitchen"
output_dir="output/ADE20K_compositional_kitchen_5_concepts"
learnable_property="object,object,object,object,object"


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
 --learnable_property ${learnable_property}  \
 --data "imagenet" --checkpointing_steps 500 \
 --mse_coeff 1 --seed 1 \
 --add_weight_per_score \
 --use_conj_score --init_weight 0.2 \
 --validation_step 500 \
 --num_iters_per_image 100 --num_images_per_class 5 \