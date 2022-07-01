import os
from sentence_transformers import SentenceTransformer, util

from config import EMBEDDINGS_FILE, EMBEDDER
import os
import pickle
# import pandas as pd
#
# import numpy as np
# from pulearn import ElkanotoPuClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EMBEDDER.load_state_dict(torch.load(EMBEDDINGS_FILE), map_location=torch.device('cpu'))
# Or you can move the loaded model into the specific device
EMBEDDER.to(device)
from flask import render_template, request, Flask
from flask import jsonify

from data_set_creation import read_embeddings

# file = open(EMBEDDINGS_FILE, 'rb')
# embeddings = pickle.load(file)
app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Make a prediction on string with 3 sentences.
    :return: a result of prediction as a json object with top100 similar
    """
    # data = request.args.get(
    data = "hello"
    #
    # df,x_data, y_labeled, y_positive = create_nn_db(corpus_embeddings,query_embeddings)
    # predict_proba,y_predict,prediction, pu_prdiction = model_train(x_data,y_labeled)
    # evaluate_results(y_positive, y_predict)
    # indxs = np.where((pu_prdiction == prediction) == True)[0]
    # res = jnsonfy(df[["text","score"]].sort_values("score", ascending=False).iloc[indxs][:100])
    return jsonify(data)


if __name__ == '__main__':
    corpus_embeddings = read_embeddings('/home/gal/PycharmProjects/CelebriteHW/data_models/corpus_embeddings.pickle')
    app.run(host='127.0.0.9', port=4455, debug=True)
