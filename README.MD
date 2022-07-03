In this project we implement an algorithmic pipeline intended to train a model with minimum intervention.
Given a Dataset, a new category name and few positive sentences as an input, train a classifier as described below.
### Example
Let’s say we want to create a new topic model that identifies violence, and we have unlabelled data for this new topic.
We came up few relevant phrases and keywords such as:
```
“they were kicking him in the yard” 
“we got involved in a fight”
“there was blood everywhere”
```
and we wish to train a model that will classify similar texts as positive for violence.
<br>The main challenge of the task is to find (automatically) relevant positive and negative samples from the unlabelled data, to construct a small training dataset.<br> For example, you can select some positive examples by using a Nearest Neighbours database.
<br>After training a model on this small training set and classifying the data (in this case, twitter sentiment analysis dataset – but we don’t use the existing labels, as we want to create a new topic model) we want the model to
classify the following tweets as positive:
```
"@huntermoore I don't want him to ever punch me.”
"I can't sleep. There's a fight outside. How inconsiderate. I wanna go SLEEP!"
"@JooceGossip Wow.... I wouldn't want to be w/ a man that would hit me in any way! I hope she doesn't go back to him!"
```

# Project run  descriptions
### Flask
This is a flask application starting from Step 4. To run it locally use `python app.py`  command from command line. 
<br>To pass a request use a browser or Postman application. The format of the URL to pass a request is:
```python
http://127.0.0.1:5000/predict?twitts=["they were kicking him in the yard", "we got involved in a fight", "there was blood everywhere"]
or
http://127.0.0.1:5000/predict?twitts=[%22they%20were%20kicking%20him%20in%20the%20yard%22,%20%22we%20got%20involved%20in%20a%20fight%22,%22there%20was%20blood%20everywhere%22]
```
<br>You should have a dataset with embeddings in the `/data_models` folder.
<br>Before Step 5 you can create a data set with enbeddings via googlecolab notebook.
All code can be found in GoogleCollab notebook with some explanations:
```python
https://colab.research.google.com/drive/1MgzivO1sEkSFk-FYnGurVKysNu27popA?usp=sharing
```
1. To download twitter sentiment analysis dataset from Kaggle we used Kaggle API.
<br>You can find it in the first googlecollab cells. Make sure you have your personal `kaggle.json` document. How to do it you can read [here](https://forum.jovian.ai/forum/t/what-is-kaggle-api-key-how-to-get-my-kaggle-key/17721).
<br> first step will load and unzip the dataset in a temporary folder on googlecollab.
2. The function `get_dataset(path_to_dataset)` in `dataset_creation.py` Loads the data (using only the text column).
3. From Step 2 we have a corpus dataset, now we can Extract features for each sample with `extract_embeddings()`.
<br> Inside it use SentenceTransformers from Hugging Face and distilled Microsoft model with embeddings size of 384 tokens.
4. After this step you able to download the embeddings data set locally (2.4Gb) and place it into `/data_models` folder. 
<br>This dataset contains embeddings + cosine similarity scores +  twitts.

<br>***Here starts the flask application run***

5. The next step is to create Nearest neighbour database/dataset with `create_nn_db(corpus_embeddings, query_embeddings)` function. It will label 25% of true positives samples with 1 and the rest 75% of samples will be unknown and marked with -1.
<br>It will call `get_samples()` function to sample positive example using the nearest neighbour DB (Based on input sentences) and negative examples
6. On this Step we Train an SVC model for ElkanotoPuClassifier from [pulearn](https://pulearn.github.io/pulearn/) library  leveraging the untagged data.
7. Evaluation is done in function `evaluate_results()` with regular scikit-learn library: F1, ROC AUC, Recall and Precision scores. <br> It will be visible in the `log_file.txt`.
8. The Response will Show top-100 most similar for input sentences results. 

Every operation and calculation results will be logged in the `log_file.txt` in the project directory.

If you want to read more about Positive Learning please read  the research papers below:
```
https://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Mordelet2013bagging.pdf
https://cseweb.ucsd.edu/~elkan/posonly.pdf
```

The code examples
```python
https://github.com/roywright/pu_learning
https://github.com/aldro61/pu-learning
https://github.com/AdityaAS/pu-learning
```