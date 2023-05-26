#!/bin/bash

python inference.py --model_path output/5_class_compositional_score_1/ --prompts "a photo of <t1>" "a photo of <t2>" "a photo of <t3>" "a photo of <t4>" "a photo of <t5>" --num_images 64 --bsz 8