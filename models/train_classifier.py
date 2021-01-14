import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Read data from SQLITE database and split them into arrays ready for machine learning
    PARAMETERS:
        database_filepath: SQLite database where disaster response data is stored
    RETURNS:
        X: 1D array explanatory variable
        Y: 2D array response variable
        category_names: list of disaster categories
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('select * from disaster_response',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(Y.columns.values)

    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and clean text messages
    PARAMETERS:
        text: raw text messages 
    RETURNS:
        clean_words: list of clean tokens
    """
    # normalize case,remove punctuation 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer  = WordNetLemmatizer()
    stop_words = stopwords.words("english")    
    clean_words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]    

    return clean_words

def build_model():
    """
    Build machine learning model using pipeline
    PARAMETERS:
        None
    RETURNS:
        Classifier machine learning model
    """
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predict and generate classification report
    PARAMETERS:
        model: classifier machine learning model
        X_test: test data for explanatory variable
        Y_test: Ground truth (correct) target values used for validation
        category_names:
    RETURNS:
        None
    """

    # predict on test data
    y_pred = model.predict(X_test)  

    # print classification report

    for i, val in enumerate(category_names):
        print("Classification Report for '{}' \n".format(val))
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))
    
    
def save_model(model, model_filepath):
    """
    Save model as pickle file
    PARAMETERS:
        model: machine learning model
        model_filepath: pickle filepath where model is saved
    RETURNS:
        None
    """
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