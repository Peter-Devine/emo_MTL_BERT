#!/bin/bash -l

#$ -S /bin/bash

#$ -N Maj_Random_Baseline_ELEP

#$ -l h_rt=0:10:0

#$ -l mem=4G

module unload compilers mpi
module load compilers/gnu/4.9.2
module load python3/recommended

pip install pandas os argparse --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/OutputMajAndRandomBaseline.py --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/EmotionLinesEmotionPush --label_name=emotion --results_path=$1/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/ELEP
