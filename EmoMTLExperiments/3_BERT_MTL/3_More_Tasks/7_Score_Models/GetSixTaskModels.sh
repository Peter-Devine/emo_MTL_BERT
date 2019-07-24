for TASKSTEM in "DD" "ELFR" "SE19" "ELEP" "ES" "SE18"
do

if [ $TASKSTEM == "ES" ] || [ $TASKSTEM == "SE18" ]
then
TASKNAME=$TASKSTEM
else
TASKNAME="${TASKSTEM}Context"
fi

mkdir $1/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/6_Six_Tasks_Evaluation/ES_SE18_SE19_ELEP_ELFR_DD_results/$TASKSTEM
cp $1/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/6_Six_Tasks/ES_SE18_SE19_ELEP_ELFR_DD_results/label_map_collection.out $1/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/6_Six_Tasks_Evaluation/ES_SE18_SE19_ELEP_ELFR_DD_results/$TASKSTEM/label_map_collection.out
cp $1/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/6_Six_Tasks/ES_SE18_SE19_ELEP_ELFR_DD_results/$TASKNAME/logs/*/*/model_state_dict.pkl $1/EmoMTLExperiments/3_BERT_MTL/3_More_Tasks/7_Score_Models/6_Six_Tasks_Evaluation/ES_SE18_SE19_ELEP_ELFR_DD_results/$TASKSTEM/model_state_dict.pkl
done
