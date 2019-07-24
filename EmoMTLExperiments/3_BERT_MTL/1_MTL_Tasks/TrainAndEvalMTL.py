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
parser.add_argument('--model_name', default="mmtl_BERT_model", help='Name of the saved model')
parser.add_argument('--num_pretraining_epochs', default="1", help='How many epochs to run pretraining for')
parser.add_argument('--num_finetune_epochs', default="1", help='How many epochs to run fine-tuning for')
parser.add_argument('--input_bert_path', default="", help='Path of BERT model if model is built on non-vanilla pre-training')
parser.add_argument('--output_bert_path', default="", help='Path of BERT model outputted after training')

args = parser.parse_args()
MAX_SEQ_LENGTH = int(args.max_seq_length)
BATCH_SIZE = int(args.batch_size)
DATA_PATH = args.data_path
TASK_LIST = args.task_list.split(",")
RESULTS_OUTPUT = args.results_output
MODEL_NAME = args.model_name
NUM_PT_EPOCHS = int(args.num_pretraining_epochs)
NUM_FT_EPOCHS = int(args.num_finetune_epochs)    
INPUT_BERT_PATH = args.input_bert_path
OUTPUT_BERT_PATH = args.output_bert_path


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

            if label_column_name in headers and not datum[headers.index(label_column_name)] in list(label_map.keys()):
                # We need to add one to this as MeTaL does not accept 0 as a valid label
                label_map[datum[headers.index(label_column_name)]] = len(list(label_map.values())) + 1

            # We tokenize our initial data
            tokens_a = tokenizer.tokenize(datum[headers.index(text_column_name)])
            if context_column_name == None:
                tokens_b = []
            else:
                tokens_b = tokenizer.tokenize(datum[headers.index(context_column_name)])

            if label_column_name in headers:
                label = label_map[datum[headers.index(label_column_name)]]
            else:
                label = 0

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
    "SST": {
        "file_path": os.path.join(DATA_PATH, "SST-2"),
        "text_column_name": "sentence",
        "label_column_name": "label",
        "context_column_name": None,
    },
}

database_tasks = {}

for task in TASK_LIST:
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

label_map_collection = {}

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
    label_map_collection[task_name] = train_inputs[3]


input_module = BertModel.from_pretrained('bert-base-uncased')
input_module.config.max_position_embeddings = MAX_SEQ_LENGTH

if INPUT_BERT_PATH:
    input_module.load_state_dict(torch.load(INPUT_BERT_PATH))

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
        task_name = database_task_name
    
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
    
        payload_train = Payload.from_tensors(payload_train_name, train_X, train_Y, task_name, "train", batch_size=batch_size)
        payload_dev = Payload.from_tensors(payload_dev_name, dev_X, dev_Y, task_name, "valid", batch_size=batch_size)
        payload_test = Payload.from_tensors(payload_test_name, test_X, test_Y, task_name, "test", batch_size=batch_size)
    
        payloads.append(payload_train)
        payloads.append(payload_dev)
        payloads.append(payload_test)
    return payloads

payloads = create_payloads(database_tasks)

model = MetalModel(tasks, verbose=False)

os.environ['METALHOME'] = RESULTS_OUTPUT

trainer = MultitaskTrainer()
trainer.config["checkpoint_config"]["checkpoint_every"] = 1
trainer.config["checkpoint_config"]["checkpoint_best"] = True

def accuracy(metrics_hist):
    metrics_agg = {}
    dev_accuracies = []
    for key in sorted(metrics_hist.keys()):
        if "accuracy" in key and "dev" in key:
            dev_accuracies.append(metrics_hist[key])

    average_accuracy = sum(dev_accuracies)/len(dev_accuracies)

    metrics_agg["model/dev/all/accuracy"] = average_accuracy

    return metrics_agg

def create_train_config(n_epochs):
    train_config = {
        "log_every": 1,
        "checkpoint_config": {
            "checkpoint_metric": "model/dev/all/accuracy",
            "checkpoint_metric_mode": "max",
        },
        "progress_bar": False,
        "optimizer_config":{
            "optimizer": "sgd",
            "optimizer_common": {"lr": 0.005},
            "sgd_config": {"momentum": 0.01},
        },
        "metrics_config":{
            "aggregate_metric_fns": [accuracy]
        },
        "n_epochs": n_epochs
    }
    return train_config

if not os.path.exists(RESULTS_OUTPUT):
    os.mkdir(RESULTS_OUTPUT)

with open(os.path.join(RESULTS_OUTPUT, "label_map_collection.out"), "wb") as fp:
    pickle.dump(label_map_collection, fp)

pretraining_results_path = os.path.join(RESULTS_OUTPUT, "pretraining")

if not os.path.exists(pretraining_results_path):
    os.mkdir(pretraining_results_path)

scores = trainer.train_model(
    model, 
    payloads,
    results_path=pretraining_results_path,
    log_every=1,
    checkpoint_config={
        "checkpoint_metric": "model/dev/all/accuracy",
        "checkpoint_metric_mode": "max",
        "checkpoint_dir": os.path.join(pretraining_results_path, "checkpoints")
    },
    writer="json",
    writer_config={
        "log_dir": os.path.join(pretraining_results_path, "logs"),
        "run_name": "pretraining"
    },
    progress_bar=False,
    optimizer_config={
        "optimizer": "sgd",
        "optimizer_common": {"lr": 0.005},
        "sgd_config": {"momentum": 0.01},
    },
    metrics_config={
        "aggregate_metric_fns": [accuracy]
    },
    n_epochs=NUM_PT_EPOCHS,
    verbose=True
)

if OUTPUT_BERT_PATH:
    torch.save(input_module.state_dict(), os.path.join(OUTPUT_BERT_PATH, "pretrained_bert.out"))

print(sys.getrecursionlimit())

task_fine_tune_results_paths = {}

for key in database_tasks.keys():
    task_fine_tune_results_path = os.path.join(RESULTS_OUTPUT, key)
    task_fine_tune_results_paths[key] = task_fine_tune_results_path
    if not os.path.exists(task_fine_tune_results_path):
        os.mkdir(task_fine_tune_results_path)

for key in database_tasks.keys():
    trainer = MultitaskTrainer()
    trainer.config["checkpoint_config"]["checkpoint_every"] = 1
    trainer.config["checkpoint_config"]["checkpoint_best"] = True

    single_task_payloads = create_payloads({key: database_tasks[key]})
    
    fine_tune_model = copy.deepcopy(model)

    scores = trainer.train_model(
    fine_tune_model,
    single_task_payloads,
    results_path=task_fine_tune_results_paths[key],
    log_every=1,
    checkpoint_config={
        "checkpoint_metric": "model/dev/all/accuracy",
        "checkpoint_metric_mode": "max",
        "checkpoint_dir": os.path.join(task_fine_tune_results_paths[key], "checkpoints")
    },
    writer="json",
    writer_config={
        "log_dir": os.path.join(task_fine_tune_results_paths[key], "logs"),
        "run_name": key
    },
    progress_bar=False,
    optimizer_config={
        "optimizer": "sgd",
        "optimizer_common": {"lr": 0.005},
        "sgd_config": {"momentum": 0.01},
    },
    metrics_config={
        "aggregate_metric_fns": [accuracy]
    },
    n_epochs=NUM_FT_EPOCHS,
    verbose=True
    )
