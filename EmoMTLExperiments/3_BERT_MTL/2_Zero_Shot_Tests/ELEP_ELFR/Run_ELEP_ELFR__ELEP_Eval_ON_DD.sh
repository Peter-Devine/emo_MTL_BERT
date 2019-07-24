#!/bin/bash -l

#$ -S /bin/bash

#$ -N EVAL_ELEP_ELFR__ELEP_ON_DD

#$ -l h_rt=1:0:0

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

/usr/bin/time --verbose python $1/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ZeroShotTest.py --max_seq_length=100 --batch_size=32 --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/ --task_list=ELEPContext,ELFRContext --results_output=$1/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ELEP_ELFR/results_ELEP_ON_DD --target_task=DDContext --inference_task=ELEPContext --emotions_map=neutral!0,surprise!6,anger!1,fear!3,non-neutral!0,joy!4,sadness!5,disgust!2 --model_path=$1/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ELEP_ELFR/model_state_dict.pkl --label_map_path=$1/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ELEP_ELFR/label_map_collection.out
