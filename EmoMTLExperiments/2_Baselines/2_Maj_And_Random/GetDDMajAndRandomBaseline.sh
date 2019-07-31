#!/bin/bash -l

#$ -S /bin/bash

#$ -N Maj_Random_Baseline_DD

#$ -l h_rt=0:10:0

#$ -l mem=4G





pip install pandas os argparse --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/OutputMajAndRandomBaseline.py --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/DailyDialogue --label_name=emotion --results_path=$1/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/DD
