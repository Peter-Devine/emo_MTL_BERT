import argparse
import os
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

parser=argparse.ArgumentParser()
parser.add_argument('--data_path', default="../1_Create_datasets/data/DailyDialogue", help='Path in which the train.tsv, dev.tsv and test.tsv files that you wish to investigate are in')
parser.add_argument('--label_name', default="emotion", help='Name of the column in which the label is')
parser.add_argument('--results_path', default="./results", help='Path in which results will be exported')

args = parser.parse_args()
DATA_PATH = args.data_path
LABEL_NAME = args.label_name
RESULTS_PATH = args.results_path

splits = ["train", "dev", "test"]

majority_class_name = "Majority class"
random_class_name = "Random class"

metrics = {
    majority_class_name: {},
    random_class_name: {},
}

for split in splits:
    data_frame = pd.read_csv(os.path.join(DATA_PATH, split + ".tsv"), sep="\t")
    
    if not LABEL_NAME in data_frame.columns:
        continue

    gold_labels = data_frame[LABEL_NAME]

    majority_class_predictions = [gold_labels.mode()[0]]*len(gold_labels)

    random_class_predictions = random.choices(gold_labels.unique(), k=len(gold_labels))

    predictions = {
        majority_class_name: majority_class_predictions,
        random_class_name: random_class_predictions
    }

    for prediction_type in predictions.keys():
        
        target_predictions = predictions[prediction_type]

        accuracy = accuracy_score(gold_labels, target_predictions)

        macro_f1 = f1_score(gold_labels, target_predictions, average='macro')

        micro_f1 = f1_score(gold_labels, target_predictions, average='micro')

        per_class_f1 = f1_score(gold_labels, target_predictions, average=None)

        intersection_labels = sorted(list(set(list(gold_labels)) & set(target_predictions)))
        
        if not "Accuracy" in metrics[prediction_type].keys():
            metrics[prediction_type]["Accuracy"] = {}
            metrics[prediction_type]["F1 (macro)"] = {}
            metrics[prediction_type]["F1 (micro)"] = {}
            metrics[prediction_type]["F1 per class"] = {}
            metrics[prediction_type]["Intersection list"] = {}

        metrics[prediction_type]["Accuracy"][split] = accuracy
        metrics[prediction_type]["F1 (macro)"][split] = macro_f1
        metrics[prediction_type]["F1 (micro)"][split] = micro_f1
        metrics[prediction_type]["F1 per class"][split] = per_class_f1
        metrics[prediction_type]["Intersection list"][split] = intersection_labels

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

f = open(os.path.join(RESULTS_PATH, "dataset_outputs.txt"), "w")
f.write("Baseline outputs:\n")
for primary_key in metrics.keys():
    f.write("\t"+primary_key+":\n")
    for secondary_key in metrics[primary_key].keys():
        f.write("\t\t"+secondary_key+":\n")
        for tertiary_key in metrics[primary_key][secondary_key].keys():
            f.write("\t\t\t"+tertiary_key+":\n")
            f.write("\t\t\t\t"+str(metrics[primary_key][secondary_key][tertiary_key])+":\n")
        
        
    f.write("\n\n")
f.close()
