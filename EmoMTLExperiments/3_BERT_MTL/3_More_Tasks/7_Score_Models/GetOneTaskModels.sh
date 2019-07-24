for TASKSTEM in "ES" "SE18"
do

mkdir $1/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/1_Singular_Tasks_Evaluation/${TASKSTEM}_results
cp $1/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/1_Singular_Tasks/${TASKSTEM}_results/label_map_collection.out $1/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/1_Singular_Tasks_Evaluation/${TASKSTEM}_results/label_map_collection.out
cp $1/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/1_Singular_Tasks/${TASKSTEM}_results/$TASKSTEM/logs/*/*/model_state_dict.pkl $1/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/1_Singular_Tasks_Evaluation/${TASKSTEM}_results/model_state_dict.pkl
done
