# import libraries

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', "stopwords","maxent_ne_chunker", "words"])
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, ne_chunk
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    
    """
    Function loads data from an sql database)
    
    Input -->   databae_filepath: file path of the database where the data stored.
    
    Output -->  X : variable that contains messages; type is array.
                Y : variable that contains categories; type is array.
                category_names : name of the categories
    """
    
    # load data from database into data frame "df"
    engine = create_engine('sqlite:///' + database_filepath)
    #table = os.path.basename(database_filepath).replace(".db","_table")
    df = pd.read_sql_table("table", engine)
    
    # Create X and y variables from df
    X = df.message.values  
    Y = df.iloc[:,4:].values
    
    # Get name of the categories
    category_names = df.iloc[:,4:].columns
    
    
    return X, Y, category_names

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
    

def build_model():
    
    # create pipeline  
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("nlp_pipeline", Pipeline([
                ("vect",CountVectorizer()),
                ("tfidf",TfidfTransformer())
            ])),

             ("text_len_ext", text_length_extractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))         
    ])

    # create parameters for grid search
    parameters = {
        "classifier__estimator__learning_rate": [0.7],
        "features__nlp_pipeline__vect__ngram_range": [(1,2)], 
        "features__nlp_pipeline__vect__max_df": [0.5],
        "features__nlp_pipeline__tfidf__norm": ["l2"], 
        "features__nlp_pipeline__tfidf__use_idf": [True], 
        "features__nlp_pipeline__tfidf__sublinear_tf":[True] 
    }
    
    # create model with gridsearch 
    model = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', n_jobs=-1)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Function displays the evaluation metrics.
    
    Input -->   model: Classification model which is the output of build_model() function
                X_test: Test data which includes text messages 
                Y_test: True categories corresponding X_test features
                category_names: labels which is the output of load_data() function
    """
    
    # get best parameters of model
    best_parameters = model.best_params_
    
    # get predicted values for X_test
    Y_pred_test = model.predict(X_test)
    
    # get overall accuracy 
    accuracy = (Y_pred_test == Y_test).mean()
    
    # get test score of model
    test_score = model.score(X_test, Y_test)
    
    # get classification report of model
    cls_report = classification_report(Y_test, Y_pred_test, target_names = category_names)
    
    print("Best parameters of model:\n", best_parameters)
    print("\nAccuracy of model:\n", accuracy)
    print("\nScore of model with test data:\n", test_score)
    print("\nClassification report of model:\n", cls_report)
    
    


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()