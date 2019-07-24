#!/bin/bash -l

#$ -S /bin/bash

#$ -N ELEP_Stats

#$ -l h_rt=0:10:0

#$ -l mem=4G

module unload compilers mpi
module load compilers/gnu/4.9.2
module load python3/recommended


pip install pandas os argparse --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/1_Datasets/2_Display_datasets/OutputDatasetStats.py --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/EmotionLinesEmotionPush --label_name=emotion --results_path=$1/EmoMTLExperiments/1_Datasets/2_Display_datasets/results/ELEP
