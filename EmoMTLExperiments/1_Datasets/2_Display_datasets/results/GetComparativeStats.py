import os
import pandas as pd
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--output_path', default="", help='Where should comparative graph output to')
parser.add_argument('--input_path', default="", help='Where should dataset data be read from')

args = parser.parse_args()
OUTPUT_PATH = args.output_path
INPUT_PATH = args.input_path

dataset_list = ["DD", "ELEP", "ELFR", "ES", "SE18", "SE19"]

relative_sizes = {}

for dataset in dataset_list:
    relative_sizes[dataset] = {}
    f = open(os.path.join(INPUT_PATH, dataset, "dataset_outputs.txt"),'r')

    dataset_stats = f.read()

    train_splits = dataset_stats.split("train: ")
    dev_splits = dataset_stats.split("dev: ")
    test_splits = dataset_stats.split("test: ")

    train_size = train_splits[1].split("\n")[0].strip()
    dev_size = dev_splits[1].split("\n")[0].strip()
    test_size = test_splits[1].split("\n")[0].strip()

    relative_sizes[dataset]["Train size"] = float(train_size)
    relative_sizes[dataset]["Validation size"] = float(dev_size)
    relative_sizes[dataset]["Test size"] = float(test_size)

relative_size_dataframe = pd.DataFrame(relative_sizes).T

bar_chart = relative_size_dataframe.plot(kind='bar', stacked=True)

bar_chart_figure = bar_chart.get_figure()
bar_chart_figure.savefig(os.path.join(OUTPUT_PATH, 'Relative_Dataset_Sizes.pdf'))
