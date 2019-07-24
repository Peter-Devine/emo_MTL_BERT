cp $1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/2_Apply_Pretrained_BERT/DD_ELFR_ELEP_SE19_results/label_map_collection.out $1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/3_Score_Models/DD_ELFR_ELEP_SE19/label_map_collection.out

for TASKNAME in "DD" "ELFR" "SE19" "ELEP"
do
mkdir $1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/3_Score_Models/DD_ELFR_ELEP_SE19/${TASKNAME}
cp $1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/2_Apply_Pretrained_BERT/DD_ELFR_ELEP_SE19_results/${TASKNAME}Context/logs/*/*/model_state_dict.pkl $1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/3_Score_Models/DD_ELFR_ELEP_SE19/${TASKNAME}/model_state_dict.pkl
done


cp $1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/2_Apply_Pretrained_BERT/ES_SE18_SE19_ELEP_ELFR_DD_results/label_map_collection.out $1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/3_Score_Models/ES_SE18_SE19_ELEP_ELFR_DD/label_map_collection.out

for TASKNAME in "DD" "ELFR" "SE19" "ELEP" "ES" "SE18"
do

if [ $TASKNAME == "ES" ] || [ $TASKNAME == "SE18" ]
then
FULLTASKNAME=$TASKNAME
else
FULLTASKNAME="${TASKNAME}Context"
fi

mkdir $1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/3_Score_Models/ES_SE18_SE19_ELEP_ELFR_DD/${TASKNAME}
cp $1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/2_Apply_Pretrained_BERT/ES_SE18_SE19_ELEP_ELFR_DD_results/$FULLTASKNAME/logs/*/*/model_state_dict.pkl $1/EmoMTLExperiments/3_BERT_MTL/4_Extended_Pretraining/3_Score_Models/ES_SE18_SE19_ELEP_ELFR_DD/${TASKNAME}/model_state_dict.pkl
done
