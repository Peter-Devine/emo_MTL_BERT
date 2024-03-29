#!/bin/bash -l

#$ -S /bin/bash

#$ -N MTL_4_tasks_DD_ELFR_ELEP_SE19

#$ -l h_rt=47:0:0

#$ -l gpu=1

#$ -l mem=20G








pip install torch torchvision tqdm pytorch_pretrained_bert tensorboardX --user
pip install -e $1/EmoMTLExperiments/metal --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/TrainAndEvalMTL.py --max_seq_length=100 --batch_size=32 --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/ --task_list=ELFRContext,DDContext,ELEPContext,SE19Context --results_output=$1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/2_Apply_Pretrained_BERT/DD_ELFR_ELEP_SE19_results --input_bert_path=$1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/1_SST_Pretraining/pretrained_bert.out --model_name=DD_ELFR_ELEP_SE19 --num_pretraining_epochs=25 --num_finetune_epochs=5
