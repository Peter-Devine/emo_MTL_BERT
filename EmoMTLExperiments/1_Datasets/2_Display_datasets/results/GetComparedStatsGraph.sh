#!/bin/bash -l

#$ -S /bin/bash

#$ -N Get_Stats_Graph

#$ -l h_rt=0:10:0

#$ -l mem=1G

module unload compilers mpi
module load compilers/gnu/4.9.2
module load python3/recommended


pip install pandas os argparse matplotlib --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/1_Datasets/2_Display_datasets/results/GetComparativeStats.py --output_path=$1/EmoMTLExperiments/1_Datasets/2_Display_datasets/results --input_path=$1/EmoMTLExperiments/1_Datasets/2_Display_datasets/results
