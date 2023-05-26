#!/bin/bash

clip_logit=0.3
resnet_logit=10

python eval.py --model_path output/5_class_compositional_score_1_seed_0/ --evaluation_metric clip --class_names "geyser" "chihuahua" "chimpanzee" "shopping cart" "mosque" --logit_threshold $clip_logit
# python eval.py --model_path output/5_class_compositional_score_2_seed_0/ --evaluation_metric clip --class_names "guinea pig" "warplane" "castle" "llama" "volcano" --logit_threshold $clip_logit
# python eval.py --model_path output/5_class_compositional_score_3_seed_0/ --evaluation_metric clip --class_names "convertible" "starfish" "studio couch" "african elephant" "teddy" --logit_threshold $clip_logit
# python eval.py --model_path output/5_class_compositional_score_4_seed_0/ --evaluation_metric clip --class_names "koala" "ice bear" "zebra" "tiger" "panda" --logit_threshold $clip_logit

python eval.py --model_path output/5_class_compositional_score_1_seed_0/ --evaluation_metric resnet --class_names "geyser" "chihuahua" "chimpanzee" "shopping cart" "mosque" --logit_threshold $resnet_logit
# python eval.py --model_path output/5_class_compositional_score_2_seed_0/ --evaluation_metric resnet --class_names "guinea pig" "warplane" "castle" "llama" "volcano" --logit_threshold $resnet_logit
# python eval.py --model_path output/5_class_compositional_score_3_seed_0/ --evaluation_metric resnet --class_names "convertible" "starfish" "studio couch" "african elephant" "teddy" --logit_threshold $resnet_logit
# python eval.py --model_path output/5_class_compositional_score_4_seed_0/ --evaluation_metric resnet --class_names "koala" "ice bear" "zebra" "tiger, Panthera tigris" "giant panda" --logit_threshold $resnet_logit