#!/bin/bash -l

#$ -S /bin/bash

#$ -N MTL_2_tasks_DD_ELEP

#$ -l h_rt=47:0:0

#$ -l gpu=1

#$ -l mem=20G








pip install torch torchvision tqdm pytorch_pretrained_bert tensorboardX --user
pip install -e $1/EmoMTLExperiments/metal --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/TrainAndEvalMTL.py --max_seq_length=100 --batch_size=32 --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/ --task_list=DDContext,ELEPContext --results_output=$1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/2_Two_Tasks/DD_ELEP_results --model_name=DD_ELEP --num_pretraining_epochs=25 --num_finetune_epochs=5
