### Disaster Response Pipeline Project

### Table of Contents
1. [Project Description](#projectdescription)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Execution](#execution)
5. [Licensing, Authors, and Acknowledgements](#licensing)
 
## Project Description <a name="projectdescription"></a> 
This project has been prepared as a deliverable of [Data Science Nanodegree Program] by Udacity.

In this project, an ETL Pipeline and a Machine Leaarning Pipeline have been built to prepare data which includes mesages from major natural disasters around the world and to categorize emergency messages based on the needs communicated by the sender.

Project also includes a wep app where somenone can input a new message and get classification results in trained categories; also the disrtibution of the trained categories can be seen in some visuals in web app.

Data of Figure Eight, a company focused on creating datasets for AI applications, has been used in this project. Updated version of the data can be reached [here]. 
 
 [Data Science Nanodegree Program]: https://www.udacity.com/course/data-scientist-nanodegree--nd025
 [here]: https://appen.com/datasets/combined-disaster-response-data/

## Installation <a name="installation"></a>

Below libraries are required in execution of the codes which are supplied in this repository:<br>

* Python 3.5+
* Machine Learning Libraries: Sciki-Learn, SciPy, Pandas, NumPy 
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Data Visualization and Web App: Flask, Plotly


## File Descriptions<a name="files"></a>

1. [disaster_messages.csv] : First part of the dataset, it contains messages.
2. [disaster_categories.csv] : Second part of the dataset, it contains categories of messages.
3. [process_data.py] : It contains necessary codes to merge above two datasets, cleaning, preparing and saving into sql database.
4. [ETL Pipeline Preparation.ipynb] : Jupyter Notebook, it covers the explanation of steps of merging above two datasets, cleaning and saving into sql database.
5. [disaster_response_database.db] : SQL database file, it contains the cleaned and prepared data, as an output of process_data.py file.
6. [train_classifier.py] : It contains necessary codes to create ML Classification model and save as pickle file.
7. [classifier.pkl] : It contains ML Classification model as an output of train_classifier.py file.
8. [ML Pipeline Preparation.ipynb] : Jupyter Notebook, it covers the explanation of steps of creating ML Classification model.
9. [templates] : Folder which contains html files for web app.
9. [run.py] : Codes for Flask web application


[disaster_messages.csv]: https://github.com/xlsxl/Disaster-Response-Pipeline-Project-/blob/main/data/disaster_messages.csv
[disaster_categories.csv]: https://github.com/xlsxl/Disaster-Response-Pipeline-Project-/blob/main/data/disaster_categories.csv
[process_data.py]: https://github.com/xlsxl/Disaster-Response-Pipeline-Project-/blob/main/data/process_data.py
[ETL Pipeline Preparation.ipynb]: https://github.com/xlsxl/Disaster-Response-Pipeline-Project-/blob/main/data/ETL%20Pipeline%20Preparation.ipynb
[disaster_response_database.db]: https://github.com/xlsxl/Disaster-Response-Pipeline-Project-/blob/main/data/disaster_response_database.db
[train_classifier.py]: https://github.com/xlsxl/Disaster-Response-Pipeline-Project-/blob/main/models/train_classifier.py
[classifier.pkl]: https://github.com/xlsxl/Disaster-Response-Pipeline-Project-/blob/main/models/classifier.pkl
[ML Pipeline Preparation.ipynb]: https://github.com/xlsxl/Disaster-Response-Pipeline-Project-/blob/main/models/ML%20Pipeline%20Preparation.ipynb
[templates]: https://github.com/xlsxl/Disaster-Response-Pipeline-Project-/tree/main/app/templates
[run.py]: https://github.com/xlsxl/Disaster-Response-Pipeline-Project-/blob/main/app/run.py

## Execution<a name="excetuion"></a>
1. To run ETL pipeline, copy-paste following commands in the project's directory: <br>
   "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_database.db"

2. To run ML pipeline, copy-paste following commands in the project's directory: <br>
   "python models/train_classifier.py data/disaster_response_database.db models/classifier.pkl"

3. To run web app, run the command "python run.py" in the project's directory and go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

This project has been prepared as a deliverable of [Data Science Nanodegree Program] by Udacity.<br>
Data of Figure Eight, a company focused on creating datasets for AI applications, has been used in this project. Updated version of the data can be reached [here].
