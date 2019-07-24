#!/bin/bash -l

#$ -S /bin/bash

#$ -N MTL_2_tasks_SE19_SE18

#$ -l h_rt=15:0:0

#$ -l gpu=1

#$ -l mem=20G

module unload compilers mpi
module load compilers/gnu/4.9.2
module load python3/recommended
module load cuda/9.0.176-patch4/gnu-4.9.2
module load cudnn/7.4.2.24/cuda-9.0
module load tensorflow/1.12.0/gpu

pip install torch torchvision tqdm pytorch_pretrained_bert tensorboardX --user
pip install -e $1/EmoMTLExperiments/metal --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/TrainAndEvalMTL.py --max_seq_length=100 --batch_size=32 --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/ --task_list=SE19Context,SE18 --results_output=$1/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/2_Two_Tasks/SE19_SE18_results --model_name=SE19_SE18 --num_pretraining_epochs=25 --num_finetune_epochs=5
