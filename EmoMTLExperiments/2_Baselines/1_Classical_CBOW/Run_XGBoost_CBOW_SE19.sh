#!/bin/bash -l

#$ -S /bin/bash

#$ -N CBOW_SE19

#$ -l h_rt=8:0:0

#$ -l gpu=0

#$ -l mem=20G





pip install spacy pandas numpy matplotlib glob spacy nltk re sklearn --user
pip install xgboost --user
python -m spacy download en_core_web_sm

mkdir $1/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/word_vectors
cd $1/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/word_vectors

if [ -f "$1/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/word_vectors/glove.6B.300d.txt" ];
then
    echo "Word vectors already downloaded"
else
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B
fi

cd ..

/usr/bin/time --verbose python $1/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/XGBoost_CBOW_Baseline.py --x_column=dialogue --y_column=emotion --data_path=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/SE19/ --output_path=$1/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/SE19_results --word_vector_txt_path=$1/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/word_vectors/glove.6B.300d.txt
