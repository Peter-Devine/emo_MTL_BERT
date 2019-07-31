#!/bin/bash -l

#$ -S /bin/bash

#$ -N EVAL_DD_ELFR_ELEP__DD

#$ -l h_rt=1:0:0

#$ -l gpu=1

#$ -l mem=10G








pip install torch torchvision tqdm pytorch_pretrained_bert tensorboardX --user
pip install -e $1/EmoMTLExperiments/metal --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ZeroShotTest.py --max_seq_length=100 --batch_size=32 --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/ --task_list=DDContext,ELFRContext,ELEPContext --results_output=$1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/DD_ELFR_ELEP/results_DD --target_task=DDContext --inference_task=DDContext --model_path=$1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/DD_ELFR_ELEP/DD/model_state_dict.pkl --label_map_path=$1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/DD_ELFR_ELEP/DD/label_map_collection.out
