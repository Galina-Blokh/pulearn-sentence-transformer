import os

from sentence_transformers import SentenceTransformer

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PATH_TO_SAVE = PROJECT_PATH + "/data_models"
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
DATA_SET = PATH_TO_SAVE + '/training.1600000.processed.noemoticon.csv'
EMBEDDINGS_CORPUS = PATH_TO_SAVE + '/corpus_embeddings.pt'
QUERY_EMBEDDINGS = PATH_TO_SAVE + '/query_embeddings.pt '
# Query sentences:
QUERIES = ["they were kicking him in the yard", "we got involved in a fight", "there was blood everywhere"]
LOG_FILE = 'log_file.txt'
NEGATIVE_THRESHOLD = -0.166
POSITIVE_THRESHOLD = .35
