#!/bin/bash -l

#$ -S /bin/bash

#$ -N create_datasets

#$ -l h_rt=1:0:0

cd $1/EmoMTLExperiments

mkdir $1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository
mkdir $1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data

cd $1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository

mkdir SST_2
cd SST_2

wget -O ./SST.zip "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8"

unzip SST

rm SST-2/original -r

cp -r SST-2 ../../data/

cd ..

mkdir DailyDialogue
cd DailyDialogue/

wget http://yanran.li/files/ijcnlp_dailydialog.zip
unzip ijcnlp_dailydialog.zip
rm ijcnlp_dailydialog.zip

cd ijcnlp_dailydialog
unzip test.zip
rm test.zip
unzip train.zip
rm train.zip
unzip validation.zip
rm validation.zip

cd ..
git clone https://github.com/Peter-UCL/DailyDailogueConverter.git
cd DailyDailogueConverter/
python DailyDialogueConverter.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/DailyDialogue/ijcnlp_dailydialog --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/DailyDialogue --turns=2 --separator=[SEP]

cd ../..

mkdir EmotionLinesFriends
cd EmotionLinesFriends
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Koxs2pVSmmO_-LWDGx3uUODVHY1yNrTM' -O Friends.tar.gz

gunzip Friends.tar.gz
tar -xvf Friends.tar

rm Friends.tar

git clone https://github.com/Peter-UCL/FriendsConverter.git
cd FriendsConverter/
python Friend_dataset_cleaner.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/EmotionLinesFriends/EmotionLines/Friends/ --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/EmotionLinesFriends --turns=2 --separator=[SEP] --is_friends=True

cd ../..

mkdir EmotionLinesEmotionPush
cd EmotionLinesEmotionPush
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uVs511ahoxHiLtSGO4sY47k4osKpT3hG' -O EmotionPush.tar.gz

gunzip EmotionPush.tar.gz
tar -xvf EmotionPush.tar

rm EmotionPush.tar

git clone https://github.com/Peter-UCL/FriendsConverter.git
cd FriendsConverter/
python Friend_dataset_cleaner.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/EmotionLinesEmotionPush/EmotionLines/EmotionPush/ --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/EmotionLinesEmotionPush --turns=2 --separator=[SEP] --is_friends=False

cd ../..

mkdir SemEval2019Task3
cd SemEval2019Task3
mkdir download_data
cd download_data

wget -O train.zip https://emocontext.blob.core.windows.net/data/starterkitdata.zip
unzip -o train
rm train.zip
wget -O dev.zip https://emocontext.blob.core.windows.net/data/devsetwithlabels.zip
unzip -o dev
rm dev.zip
wget -O testing.zip https://emocontext.blob.core.windows.net/data/test.zip
unzip -o testing
rm testing.zip

cd ..
git clone https://github.com/Peter-UCL/SemEval2019Task3Converter.git
cd SemEval2019Task3Converter
python semeval_2019_task_3_converter.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/SemEval2019Task3/download_data --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/SE19 --turn=2 --separator=[SEP]

cd ../..

mkdir SEMEVAL2018Task1
cd SEMEVAL2018Task1

mkdir download_data
cd download_data

wget http://www.saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/English/EI-reg-En-train.zip
wget http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/English/2018-EI-reg-En-dev.zip
wget http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/AIT2018-TEST-DATA/semeval2018englishtestfiles/2018-EI-reg-En-test.zip

unzip 2018-EI-reg-En-dev.zip
rm 2018-EI-reg-En-dev.zip
unzip 2018-EI-reg-En-test.zip
rm 2018-EI-reg-En-test.zip
unzip EI-reg-En-train.zip
rm EI-reg-En-train.zip

cd ..

git clone https://github.com/Peter-UCL/SEMEVAL2018Task1Converter.git

cd SEMEVAL2018Task1Converter
python SEMEVAL2018Task1.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/SEMEVAL2018Task1/download_data --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/SE18

cd ../..

mkdir ISEAR
cd ISEAR

git clone https://github.com/sinmaniphel/py_isear_dataset.git

git clone https://github.com/Peter-UCL/ISEARConverter.git

cd ISEARConverter

python ISEARConverter.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/ISEAR/py_isear_dataset/ --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/ISEAR/

cd ../..

mkdir EmotionStimulus
cd EmotionStimulus

wget http://www.eecs.uottawa.ca/~diana/resources/emotion_stimulus_data/Dataset.zip
unzip Dataset.zip

git clone https://github.com/Peter-UCL/EmotionStimulusConverter.git
cd EmotionStimulusConverter

python EmotionStimulusConverter.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/EmotionStimulus/Dataset/ --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/EmotionStimulus/

cd ../..

mkdir AffectiveText
cd AffectiveText

mkdir download_data
cd download_data

wget http://web.eecs.umich.edu/~mihalcea/downloads/AffectiveText.Semeval.2007.tar.gz
tar xvzf AffectiveText.Semeval.2007.tar.gz
rm AffectiveText.Semeval.2007.tar.gz

cp AffectiveText.trial/* ./
cp AffectiveText.test/* ./

cd ..
git clone https://github.com/Peter-UCL/AffectiveTextConverter.git
cd AffectiveTextConverter/

python AffectiveTextConverter.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/AffectiveText/download_data/ --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/AffectiveText/

cd ../..

mkdir Emobank
cd Emobank

git clone https://github.com/JULIELab/EmoBank.git

git clone https://github.com/Peter-UCL/Emobank_converter.git

cd Emobank_converter
python emobank_converter.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/Emobank/EmoBank/corpus/ --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/EmoBank/

cd ../..

mkdir EMOTERA
cd EMOTERA

mkdir data_download
cd data_download

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0B4XfYaw0oBCESU1nckw1Q3RmX28' -O EMOTERA-En.tsv
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0B4XfYaw0oBCEV0t6OEZER3NXVkU' -O EMOTERA-All.tsv
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0B4XfYaw0oBCES0k2VXhOdDczbGM' -O EMOTERA-Fil.tsv

cd ..
git clone https://github.com/Peter-UCL/EMOTERAConverter.git
cd EMOTERAConverter/
python EMOTERAConverter.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/EMOTERA/data_download/ --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/EMOTERA/

cd ../..

mkdir WASSA
cd WASSA

mkdir data_download
cd data_download

wget https://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/anger-ratings-0to1.train.txt
wget https://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/fear-ratings-0to1.train.txt
wget https://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/joy-ratings-0to1.train.txt
wget https://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/sadness-ratings-0to1.train.txt

wget https://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/anger-ratings-0to1.dev.gold.txt
wget https://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/fear-ratings-0to1.dev.gold.txt
wget https://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/joy-ratings-0to1.dev.gold.txt
wget https://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/sadness-ratings-0to1.dev.gold.txt

wget https://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/anger-ratings-0to1.test.gold.txt
wget https://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/fear-ratings-0to1.test.gold.txt
wget https://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/joy-ratings-0to1.test.gold.txt
wget https://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/sadness-ratings-0to1.test.gold.txt

cd ..

git clone https://github.com/Peter-UCL/WASSA.git
cd WASSA

python WASSAConverter.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/WASSA/data_download/ --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/WASSA/

cd ../..

mkdir FBVA
cd FBVA

mkdir data_download
cd data_download
wget http://wwbp.org/downloads/public_data/dataset-fb-valence-arousal-anon.csv

cd ..

git clone https://github.com/Peter-UCL/FBVAConverter.git
cd FBVAConverter

python FacebookVAConverter.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/FBVA/data_download --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/FBVA

cd ../..

mkdir SSEC
cd SSEC

wget http://www.romanklinger.de/ssec/ssec-aggregated-withtext.zip
unzip ssec-aggregated-withtext.zip

git clone https://github.com/Peter-UCL/SSECConverter.git
cd SSECConverter/
python SSECConverter.py --input=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/raw_data_download_repository/SSEC/ssec-aggregated --output=$1/EmoMTLExperiments/1_Datasets/1_Create_datasets/data/SSEC

cd ../../..
rm raw_data_download_repository -rf
