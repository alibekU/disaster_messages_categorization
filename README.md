# disaster_messages_categorization

# Table of contents
[Purpose](#purpose)
[Data](#data)
[Project structure](#project_structure)
[Installation](#installation)
[Usage](#usage)
[Author](#author)
[Credits](#credits)
[Requirements](#requirements)


<a name="purpose"/>
# Purpose
A web application that can categorize messages into 36 related to disaster response themes (like 'medical_help', 'weather_related' and etc.) based on training data messages that were sent during disasters around the world.
The app is hosted at ...

<a name="data"/>
# Data
The training data comes from Figure Eight's (Aspen) dataset that can be found at https://appen.com/datasets/combined-disaster-response-data. It is a dataset of thousands of messages (more than 26 000) collected during natural disasters from various sources.
For each message there are 36 possible categories (like 'medical_help', 'weather_related' and etc.)

<a name="project_structure"/>
# Project structure 
TBD

<a name="installation"/>
# Installation
1. In order to install the code and deploy the app locally please download from Github: `git clone https://github.com/alibekU/disaster_messages_categorization.git`.
2. You may want to set up a new virtual environment: `python3 venv -m environment-name-of-your-choice`
3. Then, use pip to install all the needed packages: ``

<a name="usage"/>
# Usage
After downloading, go to the the 'disaster_messages_categorization' folder and:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="author"/>
# Author 
Alibek Utyubayev
Linkedin: https://www.linkedin.com/in/alibek-utyubayev-74402721/

<a name="credits"/>
# Credits
Udacity Data Scientist Nano-Degree for the project structure and starter code

<a name="requirements"/>
# Requirements
Listed in requirements.txt file:
click==7.1.2
Flask==1.1.2
itsdangerous==1.1.0
Jinja2==2.11.2
joblib==0.17.0
MarkupSafe==1.1.1
nltk==3.5
numpy==1.19.2
pandas==1.1.3
python-dateutil==2.8.1
pytz==2020.1
regex==2020.10.11
scikit-learn==0.23.2
scipy==1.5.2
six==1.15.0
SQLAlchemy==1.3.20
threadpoolctl==2.1.0
tqdm==4.50.2
Werkzeug==1.0.1

