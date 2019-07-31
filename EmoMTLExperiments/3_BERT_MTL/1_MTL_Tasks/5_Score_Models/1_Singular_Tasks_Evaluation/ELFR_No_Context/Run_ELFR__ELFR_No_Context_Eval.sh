#!/bin/bash -l

#$ -S /bin/bash

#$ -N EVAL_ELFR__ELFR_No_Context

#$ -l h_rt=1:0:0

#$ -l gpu=1

#$ -l mem=10G








pip install torch torchvision tqdm pytorch_pretrained_bert tensorboardX --user
pip install -e $1/EmoMTLExperiments/metal --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ZeroShotTest.py --max_seq_length=100 --batch_size=32 --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/ --task_list=ELFR --results_output=$1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/ELFR_No_Context/results_ELFR_No_Context --target_task=ELFR --inference_task=ELFR --model_path=$1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/ELFR_No_Context/model_state_dict.pkl --label_map_path=$1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/ELFR_No_Context/label_map_collection.out
