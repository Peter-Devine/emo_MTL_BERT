for MODEL in "DD" "ELFR" "SE19" "ELEP"
do
mkdir $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/${MODEL}
cp $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/1_Singular_Tasks/${MODEL}_results/label_map_collection.out $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/${MODEL}/label_map_collection.out
cp $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/1_Singular_Tasks/${MODEL}_results/${MODEL}Context/logs/*/*/model_state_dict.pkl $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/${MODEL}/model_state_dict.pkl
done

for MODEL in "DD" "ELFR" "SE19" "ELEP"
do
mkdir $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/${MODEL}_No_Context
cp $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/1_Singular_Tasks/${MODEL}_No_Context_results/label_map_collection.out $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/${MODEL}_No_Context/label_map_collection.out
cp $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/1_Singular_Tasks/${MODEL}_No_Context_results/$MODEL/logs/*/*/model_state_dict.pkl $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/1_Singular_Tasks_Evaluation/${MODEL}_No_Context/model_state_dict.pkl
l
done

for MODEL in "DD_ELEP" "DD_ELFR" "DD_SE19" "ELEP_ELFR" "ELEP_SE19" "ELFR_SE19"
do
mkdir $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/$MODEL
for TASK in "DD" "ELEP" "ELFR" "SE19"
do
if [[ $MODEL == *${TASK}* ]]; then
mkdir $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/$MODEL/$TASK
cp $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/2_Two_Tasks/${MODEL}_results/label_map_collection.out $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/${MODEL}/$TASK/label_map_collection.out
cp $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/2_Two_Tasks/${MODEL}_results/${TASK}Context/logs/*/*/model_state_dict.pkl $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/2_Two_Tasks_Evaluation/${MODEL}/$TASK/model_state_dict.pkl
fi
done
done

for MODEL in "DD_ELFR_ELEP" "SE19_DD_ELEP" "SE19_ELFR_DD" "SE19_ELFR_ELEP"
do
mkdir $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/$MODEL
for TASK in "DD" "ELEP" "ELFR" "SE19"
do
if [[ $MODEL == *${TASK}* ]]; then
mkdir $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/$MODEL/$TASK
cp $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/3_Three_Tasks/${MODEL}_results/label_map_collection.out $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/${MODEL}/$TASK/label_map_collection.out
cp $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/3_Three_Tasks/${MODEL}_results/${TASK}Context/logs/*/*/model_state_dict.pkl $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/3_Three_Tasks_Evaluation/${MODEL}/$TASK/model_state_dict.pkl
fi
done
done

mkdir $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/4_Four_Tasks_Evaluation/DD_ELFR_ELEP_SE19

for TASK in "DD" "ELEP" "ELFR" "SE19"
do
mkdir $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/4_Four_Tasks_Evaluation/DD_ELFR_ELEP_SE19/$TASK
cp $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/4_Four_Tasks/DD_ELFR_ELEP_SE19_results/label_map_collection.out $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/4_Four_Tasks_Evaluation/DD_ELFR_ELEP_SE19/${TASK}/label_map_collection.out
cp $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/4_Four_Tasks/DD_ELFR_ELEP_SE19_results/${TASK}Context/logs/*/*/model_state_dict.pkl $1/EmoMTLExperiments/3_BERT_MTL/1_MTL_Tasks/5_Score_Models/4_Four_Tasks_Evaluation/DD_ELFR_ELEP_SE19/${TASK}/model_state_dict.pkl
done
