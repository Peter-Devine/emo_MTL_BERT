import argparse
import os
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('--data_path', default="../1_Create_datasets/data/DailyDialogue", help='Path in which the train.tsv, dev.tsv and test.tsv files that you wish to investigate are in')
parser.add_argument('--label_name', default="emotion", help='Name of the column in which the label is')
parser.add_argument('--results_path', default="./results", help='Path in which results will be exported')

args = parser.parse_args()
DATA_PATH = args.data_path
LABEL_NAME = args.label_name
RESULTS_PATH = args.results_path

splits = ["train", "dev", "test"]

metrics = {
    "Size": {},
    "Number of labels": {},
    "Label set": {},
    "Class distribution": {},
    "Majority class accuracy": {},
    "Random class accuracy": {}
}

for split in splits:
    data_frame = pd.read_csv(os.path.join(DATA_PATH, split + ".tsv"), sep="\t")
    
    if LABEL_NAME in data_frame.columns:
        label_series = data_frame[LABEL_NAME]
    else:
        label_series = pd.Series(["0"])


    labels_list = list(sorted(list(set(label_series))))
    number_of_observations = data_frame.shape[0]

    metrics["Size"][split] = number_of_observations
    metrics["Number of labels"][split] = len(labels_list)
    metrics["Label set"][split] = labels_list
    metrics["Class distribution"][split] = label_series.value_counts()[labels_list]/ number_of_observations
    metrics["Majority class accuracy"][split] = label_series.value_counts(normalize=True)[0]
    metrics["Random class accuracy"][split] = 1 / len(labels_list)


if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

f = open(os.path.join(RESULTS_PATH, "dataset_outputs.txt"), "w")
f.write("Dataset output:\n")
for key in metrics.keys():
    f.write("\t"+key+":\n")
    for second_key in metrics[key].keys():
        if type(metrics[key][second_key]) == pd.Series:
            f.write("\t\t" + second_key + ":\n")
            for index, series_item in metrics[key][second_key].items():
                f.write("\t\t\t"+str(index)+" - {0:.2f}\n".format(series_item))
        if type(metrics[key][second_key]) == float:
            f.write("\t\t" + second_key + ": {0:.2f} \n".format(metrics[key][second_key]))
        else:
            f.write("\t\t" + second_key + ": " + str(metrics[key][second_key]) + "\n")
        
    f.write("\n\n")
f.close()
