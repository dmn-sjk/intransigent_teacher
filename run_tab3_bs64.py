#!/usr/bin python

import itertools
import subprocess
import os
from copy import deepcopy
import time


RUN_SCRIPT_DIR = '/' + os.path.join(*__file__.split('/')[:-1])
JOBS_DIR = os.path.join(RUN_SCRIPT_DIR, 'slurm')


# /* TO MODIFY ---------------------------------------------
SLURM = False
CPUS_PER_TASK = 8
MEM_PER_CPU = 2



DATADIR = '/datasets'

# the dict of datasets, in the lists inside the dict should be setting names
datasets = {
    'cifar10_c': ['continual'],
    'domainnet126': ['continual'],
    'imagenet_others': {'imagenet_r': ['continual']},
    'imagenet_c': ['continual'],
    'ccc': ['continual'],
} 

RUN_NAME = 'tab3_bs64'

RNG_SEED=1

# all params should be in a form of array, except RUN_NAME 
configs = {
    'source': {
        'RUN_NAME': RUN_NAME,
        'TEST.BATCH_SIZE': [64], 
    },
    'tent': {
        'RUN_NAME': RUN_NAME,
        'TEST.NUM_LOOPS': [20],
        'TEST.BATCH_SIZE': [64], 
    },
    'rdumb': {
        'RUN_NAME': 'rdumb_' + RUN_NAME,
        'TEST.NUM_LOOPS': [20],
        'TEST.BATCH_SIZE': [64], 
    },
    'sar': {
        'RUN_NAME': RUN_NAME,
        'TEST.NUM_LOOPS': [20],
        'TEST.BATCH_SIZE': [64], 
    },
    'eata': {
        'RUN_NAME': RUN_NAME,
        'TEST.NUM_LOOPS': [20],
        'TEST.BATCH_SIZE': [64], 
    },
    'adacontrast': {
        'RUN_NAME': RUN_NAME,
        'M_TEACHER.FROZ': [True, False],
        'TEST.NUM_LOOPS': [20],
        'TEST.BATCH_SIZE': [64], 
    },
    'cotta': {
        'RUN_NAME': RUN_NAME,
        'M_TEACHER.FROZ': [True, False],
        'TEST.NUM_LOOPS': [20],
        'TEST.BATCH_SIZE': [64], 
    },
    'rotta': {
        'RUN_NAME': RUN_NAME,
        'M_TEACHER.FROZ': [True, False],
        'TEST.NUM_LOOPS': [20],
        'TEST.BATCH_SIZE': [64], 
    },
    'petal': {
        'RUN_NAME': RUN_NAME,
        'M_TEACHER.FROZ': [True, False],
        'TEST.NUM_LOOPS': [20],
        'TEST.BATCH_SIZE': [64], 
    },
    'norm_test': {
        'RUN_NAME': RUN_NAME,
        'WANDB': [True],
        'TEST.BATCH_SIZE': [64],
    },  #
    'memo': {
        'RUN_NAME': RUN_NAME,
        'WANDB': [True],
        'TEST.BATCH_SIZE': [64], 
    }, 
}

# argument names from the config dict above to put into the RUN_NAME string 
args_to_run_name = [
    'M_TEACHER.FROZ',
    'TEST.BATCH_SIZE'
    ]

common_args = f'DATA_DIR {DATADIR} RNG_SEED {RNG_SEED}'
# TO MODIFY */ ---------------------------------------------


def main():
    if SLURM:
        if not os.path.exists(JOBS_DIR):
            os.mkdir(JOBS_DIR)
    
    
    for dataset, settings in datasets.items():
        if dataset == 'imagenet_others':
            corruption_datasets = settings.keys()
        else:
            corruption_datasets = [None]
            
        for corruption_dataset in corruption_datasets:
            _settings = deepcopy(settings)
            if corruption_dataset is not None:
                _settings = settings[corruption_dataset]

            for setting in _settings:
                perform_experiments(dataset, setting, corruption_dataset)
                
    print('Done!')
            
def run_command(command, run_name):
    if SLURM:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)

        job_folder = os.path.join(JOBS_DIR, run_name + '_' + current_time)
        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        job_file = os.path.join(job_folder, f"{run_name}.job")
        output_file = os.path.join(job_folder, f"{run_name}.out")
        error_file = os.path.join(job_folder, f"{run_name}.err")
        
        with open(job_file, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --job-name={run_name}.job\n")
            fh.writelines(f"#SBATCH --output={output_file}\n")
            fh.writelines(f"#SBATCH --error={error_file}\n")
            fh.writelines("#SBATCH --gpus=1\n")
            fh.writelines(f"#SBATCH --cpus-per-task={CPUS_PER_TASK}\n")
            fh.writelines(f"#SBATCH --mem-per-cpu={MEM_PER_CPU}G\n")

            fh.writelines(f"cd {RUN_SCRIPT_DIR}\n")
            fh.writelines("conda activate tta\n")

            fh.writelines(command)
            fh.writelines("\n")

        os.system(f"sbatch {job_file}")
    else:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, out = process.communicate()
        print(str(out, encoding='utf-8'))
        if process.returncode == 1:
            print("--ERROR--" * 20)

def perform_experiments(dataset, setting, corruption_dataset=None):
    for method, params in configs.items():

        base_run_name = ""
        tmp_params = deepcopy(params)
        if "RUN_NAME" in params.keys():
            base_run_name = tmp_params["RUN_NAME"]
            del tmp_params["RUN_NAME"]
        
        if "TEST.NUM_LOOPS" in tmp_params.keys() and dataset == 'ccc':
            print("Experiment on CCC, changed NUM_LOOPS to 1!!!")
            tmp_params["TEST.NUM_LOOPS"] = [1 for _ in range(len(tmp_params["TEST.NUM_LOOPS"]))]
        
        for params_vals in itertools.product(*tmp_params.values()):
            run_name = base_run_name
            arguments = f'--cfg cfgs/{dataset}/{method}.yaml SETTING {setting} '
            for param_name, val in zip(tmp_params.keys(), params_vals):
                arguments += param_name + ' ' + str(val) + ' '

                if param_name in args_to_run_name:
                    if len(run_name) != 0:
                        run_name = run_name + '_' + param_name.split('.')[-1].upper() + str(val)
                    else:
                        run_name = param_name.split('.')[-1].upper() + str(val)

            if len(run_name) > 0:
                arguments += 'RUN_NAME' + ' ' + run_name + ' '
                
            if corruption_dataset is not None:
                arguments += 'CORRUPTION.DATASET' + ' ' + corruption_dataset + ' '
                
            command = f"python test_time.py {arguments}{common_args}"
            
            print(command)
            
            run_command(command, run_name)
                
if __name__ == "__main__":
    main()
