# emo_MTL_BERT
Emotional MTL with BERT experiments

## How to run this repo

Before starting to run this repo, you first need the base path where you have downloaded this repo. For example, if this repo is downloaded in C:\Users\[NAME]\Downloads\emo_MTL_BERT , or in /home/[NAME]/Downloads/emo_MTL_BERT, then C:\Users\[NAME]\Downloads\emo_MTL_BERT or /home/[NAME]/Downloads/emo_MTL_BERT would be your base path, which we will denote as $EMO_MTL_BASE_PATH. Please note that these paths must include the emo_MTL_BERT folder name too in their path. This base path will be used as an argument when each script is run.

### 1. To prepare the repo by downloading the modified version of Snorkel's MeTaL package, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/Prepare.sh $EMO_MTL_BASE_PATH`

(Note the $EMO_MTL_BASE_PATH used as an argument in running the shell script)

### 2. To build the datasets needed in the experiments, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/1_Datasets/1_Create_datasets/CreateDatasets.sh $EMO_MTL_BASE_PATH`

### 3. To get the dataset statistics, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/1_Datasets/2_Display_datasets/GetDataStats.sh $EMO_MTL_BASE_PATH`

### 4. To run the majority and random class baselines, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/GetDDMajAndRandomBaseline.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/GetELEPMajAndRandomBaseline.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/GetELFRMajAndRandomBaseline.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/2_Maj_And_Random/GetSE19MajAndRandomBaseline.sh $EMO_MTL_BASE_PATH`

### 5. To run the XGBoost baselines for each dataset both with and without context, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_DD_Context.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_DD.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_ELEP_Context.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_ELEP.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_ELFR_Context.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_ELFR.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_SE19_Context.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/2_Baselines/1_Classical_CBOW/Run_XGBoost_CBOW_SE19.sh $EMO_MTL_BASE_PATH`

### 6. To train all single task models, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/1_Singular_Tasks/Run_DD.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/1_Singular_Tasks/Run_ELEP.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/1_Singular_Tasks/Run_ELFR.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/1_Singular_Tasks/Run_SE19.sh $EMO_MTL_BASE_PATH`

### 7. To train all MTL models, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/2_Two_Tasks/Run_DD_ELEP.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/2_Two_Tasks/Run_DD_ELFR.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/2_Two_Tasks/Run_DD_SE19.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/2_Two_Tasks/Run_ELEP_ELFR.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/2_Two_Tasks/Run_ELEP_SE19.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/2_Two_Tasks/Run_SE19_ELFR.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/3_Three_Tasks/Run_DD_ELFR_ELEP.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/3_Three_Tasks/Run_SE19_DD_ELEP.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/3_Three_Tasks/Run_SE19_ELFR_DD.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/3_Three_Tasks/Run_SE19_ELFR_ELEP.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/4_Four_Tasks/Run_DD_ELFR_ELEP_SE19.sh $EMO_MTL_BASE_PATH`

### 8. To evaluate both single and MTL models, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/GetModels.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/DD/Run_DD__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/ELEP/Run_ELEP__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/ELFR/Run_ELFR__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/SE19/Run_SE19__SE19_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/DD_No_Context/Run_DD__DD_No_Context_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/ELEP_No_Context/Run_ELEP__ELEP_No_Context_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/ELFR_No_Context/Run_ELFR__ELFR_No_Context_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/SE19_No_Context/Run_SE19__SE19_No_Context_Eval.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/DD_ELEP/Run_DD_ELEP__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/DD_ELFR/Run_DD_ELFR__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/DD_SE19/Run_DD_SE19__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/ELEP_ELFR/Run_ELEP_ELFR__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/ELEP_SE19/Run_ELEP_SE19__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/ELFR_SE19/Run_ELFR_SE19__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/DD_ELEP/Run_DD_ELEP__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/DD_ELFR/Run_DD_ELFR__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/DD_SE19/Run_DD_SE19__SE19_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/ELEP_ELFR/Run_ELEP_ELFR__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/ELEP_SE19/Run_ELEP_SE19__SE19_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/ELFR_SE19/Run_ELFR_SE19__SE19_Eval.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/DD_ELFR_ELEP/Run_DD_ELFR_ELEP__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_DD_ELEP/Run_SE19_DD_ELEP__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_ELFR_DD/Run_SE19_ELFR_DD__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_ELFR_ELEP/Run_SE19_ELFR_ELEP__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/DD_ELFR_ELEP/Run_DD_ELFR_ELEP__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_DD_ELEP/Run_SE19_DD_ELEP__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_ELFR_DD/Run_SE19_ELFR_DD__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_ELFR_ELEP/Run_SE19_ELFR_ELEP__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/DD_ELFR_ELEP/Run_DD_ELFR_ELEP__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_DD_ELEP/Run_SE19_DD_ELEP__SE19_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_ELFR_DD/Run_SE19_ELFR_DD__SE19_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/SE19_ELFR_ELEP/Run_SE19_ELFR_ELEP__SE19_Eval.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/4_Four_Tasks_Evaluation/DD_ELFR_ELEP_SE19/Run_DD_ELFR_ELEP_SE19__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/4_Four_Tasks_Evaluation/DD_ELFR_ELEP_SE19/Run_DD_ELFR_ELEP_SE19__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/4_Four_Tasks_Evaluation/DD_ELFR_ELEP_SE19/Run_DD_ELFR_ELEP_SE19__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/4_Four_Tasks_Evaluation/DD_ELFR_ELEP_SE19/Run_DD_ELFR_ELEP_SE19__SE19_Eval.sh $EMO_MTL_BASE_PATH`

### 9. To train 6 task model with extra tasks, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/6_Six_Tasks/Run_ES_SE18_SE19_ELEP_ELFR_DD.sh $EMO_MTL_BASE_PATH`

### 10. To evaluate the 6 task model, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/GetSixTaskModels.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/6_Six_Tasks_Evaluation/ES_SE18_SE19_ELEP_ELFR_DD_results/Run_DD_ELFR_ELEP_SE19_SE18_ES__ES_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/6_Six_Tasks_Evaluation/ES_SE18_SE19_ELEP_ELFR_DD_results/Run_DD_ELFR_ELEP_SE19_SE18_ES__SE18_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/6_Six_Tasks_Evaluation/ES_SE18_SE19_ELEP_ELFR_DD_results/Run_DD_ELFR_ELEP_SE19_SE18_ES__SE19_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/6_Six_Tasks_Evaluation/ES_SE18_SE19_ELEP_ELFR_DD_results/Run_DD_ELFR_ELEP_SE19_SE18_ES__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/6_Six_Tasks_Evaluation/ES_SE18_SE19_ELEP_ELFR_DD_results/Run_DD_ELFR_ELEP_SE19_SE18_ES__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/6_Six_Tasks_Evaluation/ES_SE18_SE19_ELEP_ELFR_DD_results/Run_DD_ELFR_ELEP_SE19_SE18_ES__ELFR_Eval.sh $EMO_MTL_BASE_PATH`

### 11. To pre-train BERT model with SST-2, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/1_SST_Pretraining/Pretrain_SST.sh $EMO_MTL_BASE_PATH`

### 12. To pre-train BERT model with SST-2, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/1_SST_Pretraining/Pretrain_SST.sh $EMO_MTL_BASE_PATH`

### 13. To train a 4 task model using SST-2 pre-trained BERT, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/Run_DD_ELFR_ELEP_SE19__On_SST_Pretrain.sh $EMO_MTL_BASE_PATH`

### 14. To evaluate the 4 task model trained using SST-2 pre-trained BERT, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/3_Score_Models/GetModels.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/3_Score_Models/DD_ELFR_ELEP_SE19/DD/Run_DD_ELFR_ELEP_SE19__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/3_Score_Models/DD_ELFR_ELEP_SE19/ELEP/Run_DD_ELFR_ELEP_SE19__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/3_Score_Models/DD_ELFR_ELEP_SE19/ELFR/Run_DD_ELFR_ELEP_SE19__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/3_Score_Models/DD_ELFR_ELEP_SE19/SE19/Run_DD_ELFR_ELEP_SE19__SE19_Eval.sh $EMO_MTL_BASE_PATH`

### 15. To run zero-shot testing on both single task and MTL models, run:

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/GetNecessaryModels.sh $EMO_MTL_BASE_PATH`

`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ELEP/Run_Zero_Shot_ELEP__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/DD_ELFR/Run_DD_ELFR__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/DD_ELFR/Run_DD_ELFR__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/DD_ELFR_ELEP/Run_DD_ELFR_ELEP__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/DD_ELFR_ELEP/Run_DD_ELFR_ELEP__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/DD_ELFR_ELEP/Run_DD_ELFR_ELEP__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ELFR/Run_Zero_Shot_ELFR__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/DD_ELEP/Run_DD_ELEP__DD_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/DD_ELEP/Run_DD_ELEP__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ELEP_ELFR/Run_ELEP_ELFR__ELFR_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ELEP_ELFR/Run_ELEP_ELFR__ELEP_Eval.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/ELEP_ELFR/Run_ELEP_ELFR__ELEP_Eval_ON_DD.sh $EMO_MTL_BASE_PATH`
`$EMO_MTL_BASE_PATH/EmoMTLExperiments/3_BERT_MTL/2_Zero_Shot_Tests/DD/Run_Zero_Shot_DD__DD_Eval.sh $EMO_MTL_BASE_PATH`

# Order of operations

To run the entire set of experiments, run the above scripts in order from 1 -> 15.
