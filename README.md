# emo_MTL_BERT
Emotional MTL with BERT experiments

## How to run this repo

Before starting to run this repo, you first need the base path where you have downloaded this repo. For example, if this repo is downloaded in C:\Users\[NAME]\Downloads\emo_MTL_BERT , or in /home/[NAME]/Downloads/emo_MTL_BERT, then C:\Users\[NAME]\Downloads\emo_MTL_BERT or /home/[NAME]/Downloads/emo_MTL_BERT would be your base path, which we will denote as EMO_MTL_BASE_PATH. Please note that these paths must include the emo_MTL_BERT folder name too in their path. This base path will be used as an argument when each script is run.

1. To prepare the repo by downloading the modified version of Snorkel's MeTaL package, run:

EMO_MTL_BASE_PATH/EmoMTLExperiments/Prepare.sh EMO_MTL_BASE_PATH

(Note the EMO_MTL_BASE_PATH used as an argument in running the shell script)

2. To build the datasets needed in the experiments, run:

EMO_MTL_BASE_PATH/EmoMTLExperiments/1_Datasets/1_Create_datasets/CreateDatasets.sh EMO_MTL_BASE_PATH

3. To get the dataset statistics, run:

EMO_MTL_BASE_PATH/EmoMTLExperiments/1_Datasets/2_Display_datasets/GetDataStats.sh EMO_MTL_BASE_PATH

4. To run the majority and random class baselines, run:

EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/GetDDMajAndRandomBaseline.sh EMO_MTL_BASE_PATH
EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/GetELEPMajAndRandomBaseline.sh EMO_MTL_BASE_PATH
EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/GetELFRMajAndRandomBaseline.sh EMO_MTL_BASE_PATH
EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/GetSE19MajAndRandomBaseline.sh EMO_MTL_BASE_PATH

5. To run the XGBoost baselines for each dataset both with and without context, run:

EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_DD_Context.sh EMO_MTL_BASE_PATH
EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_DD.sh EMO_MTL_BASE_PATH
EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_ELEP_Context.sh EMO_MTL_BASE_PATH
EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_ELEP.sh EMO_MTL_BASE_PATH
EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_ELFR_Context.sh EMO_MTL_BASE_PATH
EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_ELFR.sh EMO_MTL_BASE_PATH
EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_SE19_Context.sh EMO_MTL_BASE_PATH
EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_SE19.sh EMO_MTL_BASE_PATH
