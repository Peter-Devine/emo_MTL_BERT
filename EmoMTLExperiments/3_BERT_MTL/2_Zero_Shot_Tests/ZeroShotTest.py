# Confirm we can import from metal
import sys
sys.path.append('../../metal')
# Import custom metal package with https://stackoverflow.com/questions/30292039/pip-install-forked-github-repo
import metal
import os
import pickle
import dill
import copy
from metal.mmtl.trainer import MultitaskTrainer
from metal.mmtl.payload import Payload
from metal.mmtl import MetalModel
from metal.mmtl.task import ClassificationTask
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert import BertTokenizer, BertModel

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

# Import other dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import pandas as pd
import csv
import sys
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--max_seq_length', default="150", help='Max size of the input')
parser.add_argument('--batch_size', default="32", help='Batch size of training')
parser.add_argument('--data_path', default="./data", help='The root file path of all the datasets')
parser.add_argument('--task_list', default="DDContext,ELFRContext,SE19Context", help='A comma separated list of all tasks to include in MTL')
parser.add_argument('--results_output', default="./results", help='The file path of the training results and output model')
parser.add_argument('--target_task', default="SE19Context", help='The task name that you would like to test on')
parser.add_argument('--inference_task', default="DDContext", help='The task name that you would like to use to create inference data')
parser.add_argument('--emotions_map', default="", help='Map of emotions from inference to target, separated by commas and each item separated by exclamation points (E.g. 0!other,1!angry,2!other,3!other,4!happy,5!sad,6!other)')
parser.add_argument('--model_path', default="./pretraining_SE19_ELFR_DD.out", help='File path of model to run inference on')
parser.add_argument('--label_map_path', default="./label_map_collection.out", help='File path of label map to run inference on')


args = parser.parse_args()
MAX_SEQ_LENGTH = int(args.max_seq_length)
BATCH_SIZE = int(args.batch_size)
DATA_PATH = args.data_path
TASK_LIST = args.task_list.split(",")
RESULTS_OUTPUT = args.results_output
TARGET_TASK_NAME = args.target_task
INFERENCE_TASK = args.inference_task
EMOTIONS_MAP = args.emotions_map
MODEL_PATH = args.model_path
LABEL_MAP_PATH = args.label_map_path

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines

def create_full_BERT_input(folder_path, text_column_name, label_column_name, context_column_name=None):

    label_map = {}

    def create_BERT_inputs_from_file(file_name, text_column_name, label_column_name, context_column_name):

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        raw_tsv_list = read_tsv(file_name)
        headers = raw_tsv_list.pop(0)

        def get_BERT_input_for_example(datum):

            if not datum[headers.index(label_column_name)] in list(label_map.keys()):
                # We need to add one to this as MeTaL does not accept 0 as a valid label
                label_map[datum[headers.index(label_column_name)]] = len(list(label_map.values())) + 1

            # We tokenize our initial data
            tokens_a = tokenizer.tokenize(datum[headers.index(text_column_name)])
            if context_column_name == None:
                tokens_b = []
            else:
                tokens_b = tokenizer.tokenize(datum[headers.index(context_column_name)])
            
            label = label_map[datum[headers.index(label_column_name)]]

            # We create our input token string by adding BERT's necessary classification and separation tokens
            # We also create the segment id list which is 0 for the first sentence, and 1 for the second sentence
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if len(tokens_b) > 0:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if len(input_ids) < MAX_SEQ_LENGTH:
                # Zero-pad up to the sequence length.
                padding = [0] * (MAX_SEQ_LENGTH - len(input_ids))

                input_ids += padding
                segment_ids += padding
            else:
                input_ids = input_ids[0:MAX_SEQ_LENGTH]
                segment_ids = segment_ids[0:MAX_SEQ_LENGTH]

            return (input_ids, segment_ids, label)

        tokenized_inputs = list(map(get_BERT_input_for_example, raw_tsv_list))

        all_input_ids = torch.tensor([tokenized_input[0] for tokenized_input in tokenized_inputs], dtype=torch.long)
        all_segment_ids = torch.tensor([tokenized_input[1] for tokenized_input in tokenized_inputs], dtype=torch.long)
        all_label_ids = torch.tensor([tokenized_input[2] for tokenized_input in tokenized_inputs], dtype=torch.long)

        return (all_input_ids, all_segment_ids, all_label_ids, label_map)

    train_path = folder_path + "/train.tsv"
    dev_path = folder_path + "/dev.tsv"
    test_path = folder_path + "/test.tsv"

    train_inputs = create_BERT_inputs_from_file(train_path, text_column_name, label_column_name, context_column_name)
    dev_inputs = create_BERT_inputs_from_file(dev_path, text_column_name, label_column_name, context_column_name)
    test_inputs = create_BERT_inputs_from_file(test_path, text_column_name, label_column_name, context_column_name)

    return (train_inputs, dev_inputs, test_inputs)

class InputTask:
    train_inputs = ()
    dev_inputs = ()
    test_inputs = ()
    no_of_lables = 0
    batch_size = 0
    label_map = {}

    def __init__(self, path, batch_size, text_column_name, label_column_name, context_column_name=None):
        self.path = path
        self.batch_size = batch_size
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.context_column_name = context_column_name

task_map = {
    "DDContext": {
        "file_path": os.path.join(DATA_PATH, "DailyDialogue"),
        "text_column_name": "dialogue",
        "label_column_name": "emotion",
        "context_column_name": "context",
    },
    "DD": {
        "file_path": os.path.join(DATA_PATH, "DailyDialogue"),
        "text_column_name": "dialogue",
        "label_column_name": "emotion",
        "context_column_name": None,
    },
    "ELFRContext": {
        "file_path": os.path.join(DATA_PATH, "EmotionLinesFriends"),
        "text_column_name": "utterance",
        "label_column_name": "emotion",
        "context_column_name": "context",
    },
    "ELFR": {
        "file_path": os.path.join(DATA_PATH, "EmotionLinesFriends"),
        "text_column_name": "utterance",
        "label_column_name": "emotion",
        "context_column_name": None,
    },
    "SE19Context": {
        "file_path": os.path.join(DATA_PATH, "SE19"),
        "text_column_name": "dialogue",
        "label_column_name": "emotion",
        "context_column_name": "context",
    },
    "SE19": {
        "file_path": os.path.join(DATA_PATH, "SE19"),
        "text_column_name": "dialogue",
        "label_column_name": "emotion",
        "context_column_name": None,
    },
    "ELEP": {
        "file_path": os.path.join(DATA_PATH, "EmotionLinesEmotionPush"),
        "text_column_name": "utterance",
        "label_column_name": "emotion",
        "context_column_name": None,
    },
    "ELEPContext": {
        "file_path": os.path.join(DATA_PATH, "EmotionLinesEmotionPush"),
        "text_column_name": "utterance",
        "label_column_name": "emotion",
        "context_column_name": "context",
    },
    "SE18": {
        "file_path": os.path.join(DATA_PATH, "SE18"),
        "text_column_name": "Tweet",
        "label_column_name": "Affect Dimension",
        "context_column_name": None,
    },
    "ES": {
        "file_path": os.path.join(DATA_PATH, "EmotionStimulus"),
        "text_column_name": "text",
        "label_column_name": "emotion",
        "context_column_name": None,
    },
}



def create_database_tasks(task_list):
    database_tasks = {}
    for task in task_list:
        if not task in task_map.keys():
            raise ValueError("Specified key name not currently supported.")
        task_details = task_map[task]
        database_tasks[task] = InputTask(
            path = task_details["file_path"],
            batch_size = BATCH_SIZE,
            text_column_name = task_details["text_column_name"],
            label_column_name = task_details["label_column_name"],
            context_column_name = task_details["context_column_name"],
        )

    for task_name in database_tasks.keys():
        database_folder_path = database_tasks[task_name].path
        text_column_name = database_tasks[task_name].text_column_name
        label_column_name = database_tasks[task_name].label_column_name
        context_column_name = database_tasks[task_name].context_column_name

        train_inputs, dev_inputs, test_inputs = create_full_BERT_input(database_folder_path, text_column_name, label_column_name, context_column_name)

        database_tasks[task_name].train_inputs = train_inputs
        database_tasks[task_name].dev_inputs = dev_inputs
        database_tasks[task_name].test_inputs = test_inputs

        database_tasks[task_name].no_of_lables = len(train_inputs[3])
        database_tasks[task_name].label_map = train_inputs[3]

    return database_tasks

database_tasks = create_database_tasks(TASK_LIST)
target_task = create_database_tasks([TARGET_TASK_NAME])

input_module = BertModel.from_pretrained('bert-base-uncased')
input_module.config.max_position_embeddings = MAX_SEQ_LENGTH

tasks = []

for task_name in database_tasks.keys():
    classification_task = ClassificationTask(
        name=task_name,
        input_module=input_module,
        head_module= torch.nn.Linear(768, database_tasks[task_name].no_of_lables))

    tasks.append(classification_task)

def create_BERT_tensor(input_ids, segment_ids):
    return(torch.cat((input_ids, segment_ids), 1))

def create_payloads(database_tasks):
    payloads = []
    splits = ["train", "valid", "test"]
    for i, database_task_name in enumerate(database_tasks):
        input_task = database_tasks[database_task_name]

        payload_train_name = f"Payload{i}_train"
        payload_dev_name = f"Payload{i}_dev"
        payload_test_name = f"Payload{i}_test"

        batch_size = input_task.batch_size

        train_inputs = input_task.train_inputs
        dev_inputs = input_task.dev_inputs
        test_inputs = input_task.test_inputs

        train_X = {"data": create_BERT_tensor(train_inputs[0], train_inputs[1])}
        dev_X = {"data": create_BERT_tensor(dev_inputs[0], dev_inputs[1])}
        test_X = {"data": create_BERT_tensor(test_inputs[0], test_inputs[1])}

        train_Y = train_inputs[2]
        dev_Y = dev_inputs[2]
        test_Y = test_inputs[2]

        payload_train = Payload.from_tensors(payload_train_name, train_X, train_Y, INFERENCE_TASK, "train", batch_size=batch_size)
        payload_dev = Payload.from_tensors(payload_dev_name, dev_X, dev_Y, INFERENCE_TASK, "valid", batch_size=batch_size)
        payload_test = Payload.from_tensors(payload_test_name, test_X, test_Y, INFERENCE_TASK, "test", batch_size=batch_size)

        payloads.append(payload_train)
        payloads.append(payload_dev)
        payloads.append(payload_test)
    return payloads

payloads = create_payloads(target_task)

model = MetalModel(tasks, verbose=False)

state_dict = torch.load(MODEL_PATH)

model.load_state_dict(state_dict)

train_payload = payloads[0]
dev_payload = payloads[1]
test_payload = payloads[2]

Y_train_preds = model.predict(train_payload, task_name=INFERENCE_TASK)
Y_dev_preds = model.predict(dev_payload, task_name=INFERENCE_TASK)
Y_test_preds = model.predict(test_payload, task_name=INFERENCE_TASK)

with open(LABEL_MAP_PATH, 'rb') as pickled_label_map:
    training_tasks_label_map = pickle.load(pickled_label_map)

inference_task_label_map = training_tasks_label_map[INFERENCE_TASK]
inverse_inference_tasks_label_map = {v: k for k, v in inference_task_label_map.items()}

if EMOTIONS_MAP:
    emotions_map_list = EMOTIONS_MAP.split(",")
    emo_map = {}
    [emo_map.update({emotion.split("!")[0]:emotion.split("!")[1]}) for emotion in emotions_map_list]

    Y_train_preds_domain_adjusted = [emo_map[inverse_inference_tasks_label_map[emo]] for emo in Y_train_preds]
    Y_dev_preds_domain_adjusted = [emo_map[inverse_inference_tasks_label_map[emo]] for emo in Y_dev_preds]
    Y_test_preds_domain_adjusted = [emo_map[inverse_inference_tasks_label_map[emo]] for emo in Y_test_preds]
else:
    Y_train_preds_domain_adjusted = [inverse_inference_tasks_label_map[emo] for emo in Y_train_preds]
    Y_dev_preds_domain_adjusted = [inverse_inference_tasks_label_map[emo] for emo in Y_dev_preds]
    Y_test_preds_domain_adjusted = [inverse_inference_tasks_label_map[emo] for emo in Y_test_preds]

target_task_dataset = target_task[TARGET_TASK_NAME]
target_task_label_map = target_task[TARGET_TASK_NAME].label_map
inverse_target_task_label_map = {v: k for k, v in target_task_label_map.items()}

Y_train_gold = [inverse_target_task_label_map[int(train_y)] for train_y in target_task_dataset.train_inputs[2]]
Y_dev_gold = [inverse_target_task_label_map[int(dev_y)] for dev_y in target_task_dataset.dev_inputs[2]]
Y_test_gold = [inverse_target_task_label_map[int(test_y)] for test_y in target_task_dataset.test_inputs[2]]

train_set_accuracy = sum(pd.Series(Y_train_preds_domain_adjusted) == pd.Series(Y_train_gold))/len(Y_train_preds_domain_adjusted)
dev_set_accuracy = sum(pd.Series(Y_dev_preds_domain_adjusted) == pd.Series(Y_dev_gold))/len(Y_dev_preds_domain_adjusted)
test_set_accuracy = sum(pd.Series(Y_test_preds_domain_adjusted) == pd.Series(Y_test_gold))/len(Y_test_preds_domain_adjusted)

train_macro_f1 = f1_score(Y_train_gold, Y_train_preds_domain_adjusted, average='macro')
dev_macro_f1 = f1_score(Y_dev_gold, Y_dev_preds_domain_adjusted, average='macro')
test_macro_f1 = f1_score(Y_test_gold, Y_test_preds_domain_adjusted, average='macro')

train_micro_f1 = f1_score(Y_train_gold, Y_train_preds_domain_adjusted, average='micro')
dev_micro_f1 = f1_score(Y_dev_gold, Y_dev_preds_domain_adjusted, average='micro')
test_micro_f1 = f1_score(Y_test_gold, Y_test_preds_domain_adjusted, average='micro')

train_labels = sorted(set(Y_train_gold))
dev_labels = sorted(set(Y_dev_gold))
test_labels = sorted(set(Y_test_gold))

train_per_class_f1 = f1_score(Y_train_gold, Y_train_preds_domain_adjusted, labels=train_labels, average=None)
dev_per_class_f1 = f1_score(Y_dev_gold, Y_dev_preds_domain_adjusted, labels=dev_labels, average=None)
test_per_class_f1 = f1_score(Y_test_gold, Y_test_preds_domain_adjusted, labels=test_labels, average=None)

def get_per_class_accuracy(y_true, y_pred):
    labels = sorted(list(set(y_true)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    per_class_accuracy = cm.diagonal()

    per_class_accuracy_df = pd.DataFrame({"Label": labels, "Accuracy": per_class_accuracy})
    
    return cm, per_class_accuracy_df

train_cm, train_per_class_accuracy = get_per_class_accuracy(Y_train_gold, Y_train_preds_domain_adjusted)
dev_cm, dev_per_class_accuracy = get_per_class_accuracy(Y_dev_gold, Y_dev_preds_domain_adjusted)
test_cm, test_per_class_accuracy = get_per_class_accuracy(Y_test_gold, Y_test_preds_domain_adjusted)

train_gold_value_counts = pd.Series(Y_train_gold).value_counts()
dev_gold_value_counts = pd.Series(Y_dev_gold).value_counts()
test_gold_value_counts = pd.Series(Y_test_gold).value_counts()

train_pred_value_counts = pd.Series(Y_train_preds_domain_adjusted).value_counts()
dev_pred_value_counts = pd.Series(Y_dev_preds_domain_adjusted).value_counts()
test_pred_value_counts = pd.Series(Y_test_preds_domain_adjusted).value_counts()

print("Train accuracy: {0}".format(train_set_accuracy))
print("Dev accuracy: {0}".format(dev_set_accuracy))
print("Test accuracy: {0}".format(test_set_accuracy))

if not os.path.exists(RESULTS_OUTPUT):
    os.makedirs(RESULTS_OUTPUT)

output_eval_file = os.path.join(RESULTS_OUTPUT, "testing_accuracy.txt")

with open(output_eval_file, "w") as writer:
    writer.write("Train accuracy: {0}\n".format(train_set_accuracy))
    writer.write("Dev accuracy: {0}\n".format(dev_set_accuracy))
    writer.write("Test accuracy: {0}\n".format(test_set_accuracy))
    writer.write("\nTrain F1 (macro): {0}\n".format(train_macro_f1))
    writer.write("Dev F1 (macro): {0}\n".format(dev_macro_f1))
    writer.write("Test F1 (macro): {0}\n".format(test_macro_f1))
    writer.write("\nTrain F1 (micro): {0}\n".format(train_micro_f1))
    writer.write("Dev F1 (micro): {0}\n".format(dev_micro_f1))
    writer.write("Test F1 (micro): {0}\n".format(test_micro_f1))
    writer.write("\nTrain F1 (per class in {0}): {1}\n".format(train_labels, train_per_class_f1))
    writer.write("Dev F1 (per class in {0}): {1}\n".format(dev_labels, dev_per_class_f1))
    writer.write("Test F1 (per class in {0}): {1}\n".format(test_labels, test_per_class_f1))
    writer.write("\nTrain per class accuracy: {0}\n".format(train_per_class_accuracy))
    writer.write("Dev per class accuracy: {0}\n".format(dev_per_class_accuracy))
    writer.write("Test per class accuracy: {0}\n".format(test_per_class_accuracy))
    writer.write("\nGold train value counts: {0}\n".format(train_gold_value_counts))
    writer.write("Gold dev value counts: {0}\n".format(dev_gold_value_counts))
    writer.write("Gold test value counts: {0}\n".format(test_gold_value_counts))
    writer.write("\nTrain predictions value counts: {0}\n".format(train_pred_value_counts))
    writer.write("Dev predictions value counts: {0}\n".format(dev_pred_value_counts))
    writer.write("Test predictions value counts: {0}\n".format(test_pred_value_counts))
    writer.write("\nTrain predictions confusion matrix (same order as class accuracies): {0}\n".format(train_cm))
    writer.write("Dev predictions confusion matrix (same order as class accuracies): {0}\n".format(dev_cm))
    writer.write("Test predictions confusion matrix (same order as class accuracies): {0}\n".format(test_cm))

output_eval_file = os.path.join(RESULTS_OUTPUT, "predictions_and_labels.txt")

with open(output_eval_file, "w") as writer:
    writer.write("Train predictions: {0}".format(Y_train_preds_domain_adjusted))
    writer.write("Train labels: {0}".format(Y_train_gold))
    writer.write("Dev predictions: {0}".format(Y_dev_preds_domain_adjusted))
    writer.write("Dev labels: {0}".format(Y_dev_gold))
    writer.write("Test predictions: {0}".format(Y_test_preds_domain_adjusted))
    writer.write("Test labels: {0}".format(Y_test_gold))
