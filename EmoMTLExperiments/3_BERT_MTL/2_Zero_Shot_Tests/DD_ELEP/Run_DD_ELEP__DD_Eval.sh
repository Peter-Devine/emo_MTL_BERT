#!/bin/bash -l

#$ -S /bin/bash

#$ -N EVAL_DD_ELEP__DD

#$ -l h_rt=1:0:0

#$ -l gpu=1

#$ -l mem=20G








pip install torch torchvision tqdm pytorch_pretrained_bert tensorboardX --user
pip install -e $1/EmoMTLExperiments/metal --user

/usr/bin/time --verbose python $1/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ZeroShotTest.py --max_seq_length=100 --batch_size=32 --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/ --task_list=DDContext,ELEPContext --results_output=$1/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/DD_ELEP/results_DD --target_task=SE19Context --inference_task=DDContext --emotions_map=0!others,1!angry,2!others,3!others,4!happy,5!sad,6!others --model_path=$1/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/DD_ELEP/model_state_dict.pkl --label_map_path=$1/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/DD_ELEP/label_map_collection.out
