import json
import logging

import torch
from flask import Flask, request

from config import EMBEDDER, EMBEDDINGS_CORPUS, QUERY_EMBEDDINGS, LOG_FILE
from data_creation import read_embeddings, extract_embeddings, get_top100

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Or you can move the loaded model into the specific device
EMBEDDER.to(device)
# log-file will be created in the main dir
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Get top100 most similar twitts for list with 3 sentences.
    :return: a result of prediction as a json object with top100 similar
    """
    # get query list with sentences
    twitts = request.args.get("twitts")
    query_list = []
    for i in json.loads(twitts):
        query_list.append(i)
    logging.debug("query_list\n" + str(query_list))
    corpus_embeddings = read_embeddings(EMBEDDINGS_CORPUS)
    logging.debug("got embeddings from corpus")
    _, query_embeddings = extract_embeddings(query_list, QUERY_EMBEDDINGS)
    logging.debug("got embeddings from query")
    top100 = get_top100(corpus_embeddings, query_embeddings)
    return top100


if __name__ == '__main__':
    app.run(debug=True)
