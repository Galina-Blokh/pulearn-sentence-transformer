import os
from sentence_transformers import SentenceTransformer, util
import os
import pickle
import pandas as pd

import numpy as np
from pulearn import ElkanotoPuClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PATH_TO_SAVE = PROJECT_PATH + "/data_models"
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
DATA_SET = PATH_TO_SAVE + '/training.1600000.processed.noemoticon.csv'
EMBEDDINGS_FILE = PATH_TO_SAVE + '/corpus_embeddings.pickle'
QUERY_EMBEDDINGS =PATH_TO_SAVE + '/query_embeddings.pickle '
# Query sentences:
QUERYS = ["they were kicking him in the yard", "we got involved in a fight","there was blood everywhere"]
