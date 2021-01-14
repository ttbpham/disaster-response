# Disaster Response Pipeline Project

This project builds a disaster response pipeline where messages captured by emergency workers on the Webapp are cleaned and classified into 36 categories. The Webapp also allow visualization of the data via bar and pie charts.

## Table of Contents

* [File Descriptions](#file-description)
* [Instructions](#instructions)
* [Creator](#creators)
* [Acknowlegement](#acknowlegement)

## File Descriptions

* master.html  - main page of web app
* go.html  - classification result page of web app
* run.py  - Flask file that runs app
* disaster_categories.csv  - input file
* disaster_messages.csv  - input file
* process_data.py - load and process input files
* ETL Pipeline Preparation.ipynb - ETL pipeline notebook
* DisasterResponse.db - store clean data for ML and Webapp
* train_classifier.py - train
* classifier.pkl  - saved model
* ML Pipeline Preparation.ipynb  # ML pipeline notebook

## Instructions

You can clone Disaster Response project using command below:

```
$ git clone https://github.com/ttbpham/disaster-response.git

```
Following below steps to execute the project
*  Under the data directory, run the ETL pipeline that cleans and store the cleaned data in to SQLite database, in the command prompt type:
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
* Under the model directory, run the ML pipeline to train the classifier with GridSearchCV and save to the pickle file, in the command prompt type:
```
 python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
 ```
* Under the the app directory,run the web app, in the command prompt type:
```
python run.py
```
* Open the http://localhost:3001 to check the web app.


## Creators

Thuy Pham  - [https://github.com/ttbpham](https://github.com/ttbpham)

## Acknowlegement

This project is under the Udacity Data Science Nanodegree.
