# coding: utf-8
# This code is based on the baseline provided by ytian22 at:
# https://github.com/ytian22/Movie-Review-Classification/blob/master/word-embeddings.ipynb

import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import spacy
import string
import re
import nltk
from spacy.symbols import ORTH
import en_core_web_sm
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

parser=argparse.ArgumentParser()
parser.add_argument('--x_column', default="dialogue", help='Column name of the inputs to the XGBoost algorithm')
parser.add_argument('--y_column', default="emotion", help='Column name of the label of the XGBoost algorithm')
parser.add_argument('--data_path', default="./data", help='The root file path of all the train.tsv, dev.tsv and test.tsv files')
parser.add_argument('--output_path', default="./results", help='Results folder path')
parser.add_argument('--word_vector_txt_path', default="./word_vectors/glove.6B.300d.txt", help='Full file path of downloaded word vector file')
parser.add_argument('--context_column', default="", help='Name of context column. Leave blank if no context')

args = parser.parse_args()
X_COLUMN = args.x_column
Y_COLUMN = args.y_column
DATA_PATH = args.data_path
OUTPUT_PATH = args.output_path
WORD_VECTOR_PATH = args.word_vector_txt_path
CONTEXT_COLUMN = args.context_column

train_data = pd.read_csv(os.path.join(DATA_PATH, "train.tsv"), sep="\t")
dev_data = pd.read_csv(os.path.join(DATA_PATH, "dev.tsv"), sep="\t")
test_data = pd.read_csv(os.path.join(DATA_PATH, "test.tsv"), sep="\t")

# borrowed from fast.ai (https://github.com/fastai/fastai/blob/master/fastai/nlp.py)
re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
def sub_br(x): return re_br.sub("\n", x)

my_tok = en_core_web_sm.load()
def spacy_tok(x): return [tok.text for tok in my_tok.tokenizer(sub_br(x))]

# get stop words
nltk.download('stopwords')
from nltk.corpus import stopwords
stops=set(stopwords.words('english'))

# modified from https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb
def get_non_stopwords(review):
    """Returns a list of non-stopwords"""
    return {x:1 for x in spacy_tok(str(review).lower()) if x not in stops}.keys()

def load_word_embeddings(file):
    embeddings={}
    with open(file,'r', encoding="utf-8") as infile:
        for line in infile:
            values=line.split()
            embeddings[values[0]]=np.asarray(values[1:],dtype='float32')
    return embeddings

embeddings = load_word_embeddings(WORD_VECTOR_PATH)

def sentence_features_v2(s, embeddings=embeddings,emb_size=300):
    # ignore stop words
    words=get_non_stopwords(s)
    words=[w for w in words if w.isalpha() and w in embeddings]
    if len(words)==0:
        return np.hstack([np.zeros(emb_size)])
    M=np.array([embeddings[w] for w in words])
    return M.mean(axis=0)

if len(CONTEXT_COLUMN) > 0:
    train_data[X_COLUMN] = train_data[X_COLUMN] + " " + train_data[CONTEXT_COLUMN].fillna("")
    dev_data[X_COLUMN] = dev_data[X_COLUMN] + " " + dev_data[CONTEXT_COLUMN].fillna("")
    test_data[X_COLUMN] = test_data[X_COLUMN] + " " + test_data[CONTEXT_COLUMN].fillna("")

# create sentence vectors
x_train = np.array([sentence_features_v2(x) for x in train_data[X_COLUMN]])
x_dev = np.array([sentence_features_v2(x) for x in dev_data[X_COLUMN]])
x_test = np.array([sentence_features_v2(x) for x in test_data[X_COLUMN]])

x_combined = np.concatenate((x_train, x_dev))

ohe = OneHotEncoder(sparse=False)

ohe.fit(np.array(train_data[Y_COLUMN].astype(str)).reshape(-1, 1))

y_train = ohe.transform(np.array(train_data[Y_COLUMN].astype(str)).reshape(-1, 1))
y_dev = ohe.transform(np.array(dev_data[Y_COLUMN].astype(str)).reshape(-1, 1))
y_test = ohe.transform(np.array(test_data[Y_COLUMN].astype(str)).reshape(-1, 1))

y_combined = np.concatenate((y_train, y_dev))

clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=5))

possible_parameters = {"estimator": [XGBClassifier(n_jobs=-1, max_depth=3, n_estimators=10),
                                     XGBClassifier(n_jobs=-1, max_depth=4, n_estimators=10),
                                     XGBClassifier(n_jobs=-1, max_depth=3, n_estimators=100),
                                     XGBClassifier(n_jobs=-1, max_depth=4, n_estimators=100)]}

grid_search = GridSearchCV(clf, possible_parameters, cv=5)

grid_search.fit(x_combined, y_combined)

y_test_preds = list(ohe.inverse_transform(grid_search.predict(x_test)).reshape(1, -1)[0])
y_test_gold = list(ohe.inverse_transform(y_test).reshape(1, -1)[0])

comparison_list = pd.Series(y_test_gold) == pd.Series(y_test_preds)

accuracy = sum(comparison_list) / len(y_test_gold)
f1_score_macro = f1_score(y_test_gold, y_test_preds, average='macro')
f1_score_micro = f1_score(y_test_gold, y_test_preds, average='micro')

label_set = sorted(set(y_test_gold))

f1_score_per_class = f1_score(y_test_gold, y_test_preds, labels=label_set, average=None)

print("Accuracy: {0}".format(accuracy))
print("F1 score macro: {0}".format(f1_score_macro))
print("F1 score micro: {0}".format(f1_score_micro))

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

output_eval_file = os.path.join(OUTPUT_PATH, "results_and_predictions.txt")

with open(output_eval_file, "w") as writer:
    writer.write("Accuracy: {0}\n".format(accuracy))
    writer.write("F1 score macro: {0}\n".format(f1_score_macro))
    writer.write("F1 score micro: {0}\n".format(f1_score_micro))
    writer.write("Label set: {0}\n".format(label_set))
    writer.write("F1 score per class: {0}\n".format(f1_score_per_class))
    writer.write("Predictions:\n")
    for y_test_pred in y_test_preds:
        writer.write("{0}\n".format(y_test_pred))
    writer.write("Labels:\n")
    for y_test_label in y_test_gold:
        writer.write("{0}\n".format(y_test_label))
