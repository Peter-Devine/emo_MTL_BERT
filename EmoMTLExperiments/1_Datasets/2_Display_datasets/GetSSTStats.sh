#!/bin/bash -l

#$ -S /bin/bash

#$ -N SST_Stats

#$ -l h_rt=0:10:0

#$ -l mem=4G






pip install pandas os argparse --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/1_Datasets/2_Display_datasets/OutputDatasetStats.py --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/SST-2 --label_name=label --results_path=$1/EmoMTLExperiments/1_Datasets/2_Display_datasets/results/SST
