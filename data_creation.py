import logging
from flask import jsonify
from config import EMBEDDER, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD
from pulearn import ElkanotoPuClassifier
from sentence_transformers import util
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_dataset(path_to_dataset):
    """
    To get a twitts dataset only  text column
    :path_to_dataset: string path to your data set
    :return:list of tweets, np.array
    """
    # Corpus with example sentences
    df = pd.read_csv(path_to_dataset, delimiter=',', encoding="ISO-8859-1", header=None, usecols=[5],
                     names=['text']).drop_duplicates().dropna()
    logging.debug("twitts corpus is in dataset")
    return df['text'].values


def extract_embeddings(corpus, path_to_save):
    """
    To extract sentence embeddings using SentenceTransformer and cosine similarity
    Uses cuda and GPS (min 8Gb)
    :param path_to_save: string path where you ant to save pkl file with tweet embeddings
    :param corpus: a list or np.array of tweets
    :path_to_save: string folder path where your embeddings will be saved
    :return: string path to .pt file path with embeddings ordered as {'embeddings':corpus_embeddings,'text':corpus}
    """
    # extract
    corpus_embeddings = EMBEDDER.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
    # Save
    torch.save({'embeddings': corpus_embeddings, 'text': corpus}, path_to_save)
    logging.debug(f"embeddings saved to {path_to_save}")
    return path_to_save, {'embeddings': corpus_embeddings, 'text': corpus}


def read_embeddings(path_embeddings):
    """
    To load tweet embeddings from disk.
    :param path_embeddings: string path to file with embeddings
    :return: data set {'embeddings':corpus_embeddings,'text':corpus}
    """
    logging.debug(f"read embeddings from disk \n{path_embeddings}")
    return torch.load(path_embeddings, map_location=device)


def get_samples(corpus_embeddings, queries, is_positive=False):
    """
    To sample positive and negative tweets
    :param queries: list of strings/tweets to classify
    :param corpus_embeddings: {'embeddings':corpus_embeddings,'text':corpus}
    :param is_positive: flag to switch between positive and negative results  extraction
    :return: list of dict {"indx": i, "text": corpus[i], "score": round(s.item(), 4),
                         "embeddings": corpus_embeddings['embeddings'][i]}
    """
    cos_scores_val = []
    cos_scores_ind = []
    logging.debug(f"start iteration on query sentences for is_positive = {is_positive}")
    for query in queries['embeddings']:
        # We use cosine-similarity
        cos_scores = util.cos_sim(query, corpus_embeddings['embeddings'])[0]
        if is_positive:
            thresh = POSITIVE_THRESHOLD
            cos_scores_val.append(cos_scores[cos_scores >= thresh])
            cos_scores_ind.append((cos_scores >= thresh).nonzero())
        else:
            thresh = NEGATIVE_THRESHOLD
            cos_scores_val.append(cos_scores[cos_scores <= thresh])
            cos_scores_ind.append((cos_scores <= thresh).nonzero())
    hits = []
    logging.debug("build list of dicts with features for each twitte")
    # build list of dicts with features for each twitte
    for score, idx in zip(cos_scores_val, cos_scores_ind):
        for s, i in zip(score, idx.flatten()):
            hits.append({"indx": i.item(),
                         "text": corpus_embeddings["text"][i],
                         "score": round(s.item(), 4),
                         "embeddings": corpus_embeddings['embeddings'][i].cpu().detach().numpy()})
    return hits


def create_nn_db(corpus_embeddings, query_embeddings):
    """
    Create a Nearest Neighbour data set by sampling positive and negative classes
    :param query_embeddings: a dataframe with tweet embeddings and cosine similarity feature for query
    :param corpus_embeddings: a dataframe with tweet embeddings and cosine similarity feature for corpus
    :return: a dataset with positive and negative samples of embeddings,cosine score  and  binary labels
    """
    logging.debug("start NN DB creation")
    positive_ds = pd.DataFrame(get_samples(corpus_embeddings, query_embeddings, True))  # 4687 rows
    negative_ds = pd.DataFrame(get_samples(corpus_embeddings, query_embeddings))  # 4794 rows
    logging.debug("start labeling")
    # labeling
    positive_ds['label'] = 1
    negative_ds['label'] = 0
    # concat and shuffle
    dfs = [positive_ds, negative_ds]
    df = pd.concat(dfs, axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    # get the indices of the positives samples
    pos_ind = np.where(df.iloc[:, -1].values == 1)[0]
    # shuffle them
    np.random.shuffle(pos_ind)
    # leave just 25% of the positives marked
    pos_sample_len = int(np.ceil(0.25 * len(pos_ind)))
    logging.debug(f'Using {pos_sample_len}/{len(pos_ind)} as positives and unlabeling the rest')
    pos_sample = pos_ind[:pos_sample_len]
    # mask 75% labels
    df['class_test'] = -1
    df.loc[pos_sample, 'class_test'] = 1
    logging.debug(f'target variable:\n{df.iloc[:, -1].value_counts()}')
    # create input for pu model learning
    x_data = np.concatenate(
        (np.stack(df['embeddings'].to_numpy(), axis=0), df['score'].to_numpy().reshape(-1, 1)),
        axis=1)  # just the X
    y_labeled = df.iloc[:, -1].values  # new class (just the P & U)
    y_positive = df.iloc[:, -2].values  # original class
    logging.debug("NN DB created")
    return df, x_data, y_labeled, y_positive


def model_train(x_data, y_labeled):
    """
    Train the Support Vector classifier on unlabeled data
    :param x_data: np.array with tweet embeddings and cosine sim. score
    :param y_labeled: np.array with only 25% positives(1) and rest unlabeled (-1)
    :return: tuple (predict_proba: np.array with class probabilities
             y_predict: list of predicted labels with threshold 0.5
             prediction: np.array with labeled data predictions
             pu_prdiction: np.array with unlabeled data predictions)
    """
    logging.debug("set estimator SVC")
    estimator = SVC(
        C=10,
        kernel='rbf',
        gamma=0.4,
        probability=True,
    )
    pu_estimator = ElkanotoPuClassifier(estimator, hold_out_ratio=0.2)
    pu_estimator.fit(x_data, y_labeled)
    prediction = estimator.predict(x_data)
    pu_prdiction = pu_estimator.predict(x_data)
    logging.debug("\nComparison of estimator and PUAdapter(estimator):")
    logging.debug("Number of disagreements: {}".format(len(np.where((pu_prdiction == prediction) == False)[0])))
    print("Number of agreements: {}".format(len(np.where((pu_prdiction == prediction) == True)[0])))
    predict_proba = pu_estimator.predict_proba(x_data)
    y_predict = [1 if x > 0.5 else 0 for x in predict_proba]
    return predict_proba, y_predict, prediction, pu_prdiction


def evaluate_results(y_test, y_predict):
    """
    Print our f1_score, roc_auc_score,recall_score and precision_score
    :param y_test:  np.array with original class (y_positive)
    :param y_predict: list of predicted labels with threshold 0.5
    """
    logging.debug('Classification results:')
    f1 = f1_score(y_test, y_predict)
    logging.debug("f1: %.2f%%" % (f1 * 100.0))
    roc = roc_auc_score(y_test, y_predict)
    logging.debug("roc: %.2f%%" % (roc * 100.0))
    rec = recall_score(y_test, y_predict, average='binary')
    logging.debug("recall: %.2f%%" % (rec * 100.0))
    prc = precision_score(y_test, y_predict, average='binary')
    logging.debug("precision: %.2f%%" % (prc * 100.0))


def get_top100(corpus_embeddings, query_embeddings):
    """

    :param corpus_embeddings: tensor with corpus embeddings
    :param query_embeddings: tensor with query embeddings
    :return: list  of top100 similar to query sentences
    """
    df, x_data, y_labeled, y_positive = create_nn_db(corpus_embeddings, query_embeddings)
    _, y_predict, prediction, pu_prdiction = model_train(x_data, y_labeled)
    evaluate_results(y_positive, y_predict)
    logging.debug("extracting indexes and prediction text")
    indxs = np.where((pu_prdiction == prediction) == True)[0]
    logging.debug(str(df[["text", "score"]].sort_values("score", ascending=False).iloc[indxs][:100].values.tolist()))
    return jsonify(df[["text", "score"]].sort_values("score", ascending=False).iloc[indxs][:100].values.tolist())

