import json
import plotly
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', "stopwords","maxent_ne_chunker", "words"])
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, ne_chunk

from sklearn.base import BaseEstimator, TransformerMixin


from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as go
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Funnction tokenize the input text.
    
    Input-->    text: text to be tokenized, as string
    
    Output-->   lemmatized: List of words of cleaned text, as list
    """
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # Normalization : 
    # Replace punctuations with " " and make string lowercase
    # Replace all punctuations except apostrophes
    text = re.sub("[^a-zA-Z0-9']", " ", text.lower())
    
    # Tokenize: Split to words
    tokenized = text.split(" ")
    tokenized = [word for word in tokenized if word != "" ]
    
    # Remove stop_words
    cleaned = [word for word in tokenized if word not in stop_words]
    
    # Part of speech tagging
    tagged = nltk.pos_tag(cleaned)
    
    # Lemmatize
    lemmatized = []

    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    for word, raw_tag in tagged:
        tag = tag_dict.get(raw_tag[0].upper(), wordnet.ADV)
        lemmatized.append(lemmatizer.lemmatize(word, pos = tag))

    return lemmatized

# Create custom transformer which calculates text lenght 
class text_length_extractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        extracted_lengths = pd.Series(X).apply(lambda x:len(x))
        return pd.DataFrame(extracted_lengths)

# load data
engine = create_engine('sqlite:///../data/disaster_response_database.db')
df = pd.read_sql_table('table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extraction for categories
    category_names = df.iloc[:,4:].sum().sort_values(ascending = False).index.tolist()
    category_counts = df.iloc[:,4:].sum().sort_values(ascending = False).values.tolist()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category Names",
                    "tickangle":45
                }
            }
        }            
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()