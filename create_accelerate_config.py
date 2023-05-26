import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, required=True)
parser.add_argument("--distributed", action="store_true", default=False)
args = parser.parse_args()

dict_file = {
    'command_file': None,
    'commands': None,
    'compute_environment': 'LOCAL_MACHINE',
    'deepspeed_config': {},
    'distributed_type': 'NO',
    'downcast_bf16': 'no',
    'dynamo_backend': 'NO',
    'fsdp_config': {},
    'gpu_ids': args.gpu_id,
    'machine_rank': 0,
    'main_process_ip': None,
    'main_process_port': None,
    'main_training_function': 'main',
    'megatron_lm_config': {},
    'mixed_precision': 'no',
    'num_machines': 1,
    'num_processes': 1,
    'rdzv_backend': 'static',
    'same_network': True,
    'tpu_name': None,
    'tpu_zone': None,
}

dist_dict_file = {
    'command_file': None,
    'commands': None,
    'compute_environment': 'LOCAL_MACHINE',
    'deepspeed_config': {},
    'distributed_type': 'MULTI_GPU',
    'downcast_bf16': 'no',
    'dynamo_backend': 'NO',
    'fsdp_config': {},
    'gpu_ids': args.gpu_id,
    'machine_rank': 0,
    'main_process_ip': None,
    'main_process_port': None,
    'main_training_function': 'main',
    'megatron_lm_config': {},
    'mixed_precision': 'fp16',
    'num_machines': 1,
    'num_processes': len(args.gpu_id.split(',')),
    'rdzv_backend': 'static',
    'same_network': True,
    'tpu_name': None,
    'tpu_zone': None,
    'use_cpu': False,
}

if args.distributed:
    with open('accelerate_config.yaml', 'w') as file:
        documents = yaml.dump(dist_dict_file, file)
else:
    with open('accelerate_config.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file)
