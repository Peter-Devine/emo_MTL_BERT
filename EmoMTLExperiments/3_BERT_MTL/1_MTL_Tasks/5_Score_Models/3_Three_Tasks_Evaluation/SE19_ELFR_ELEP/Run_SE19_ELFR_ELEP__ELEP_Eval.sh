#!/bin/bash -l

#$ -S /bin/bash

#$ -N EVAL_SE19_ELFR_ELEP__ELEP

#$ -l h_rt=1:0:0

#$ -l gpu=1

#$ -l mem=10G








pip install torch torchvision tqdm pytorch_pretrained_bert tensorboardX --user
pip install -e $1/EmoMTLExperiments/metal --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ZeroShotTest.py --max_seq_length=100 --batch_size=32 --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/ --task_list=SE19Context,ELFRContext,ELEPContext --results_output=$1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_ELFR_ELEP/results_ELEP --target_task=ELEPContext --inference_task=ELEPContext --model_path=$1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_ELFR_ELEP/ELEP/model_state_dict.pkl --label_map_path=$1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_ELFR_ELEP/ELEP/label_map_collection.out
