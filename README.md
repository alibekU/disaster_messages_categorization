# disaster_messages_categorization

# Table of contents
- [Purpose](#purpose)
- [Data](#data)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Author](#author)
- [Credits](#credits)
- [Requirements](#requirements)


# Purpose
A web application that can categorize messages into 36 related to disaster response themes (like 'medical_help', 'weather_related' and etc.) based on training data messages that were sent during disasters around the world.
The app is hosted at ...

# Data
The training data comes from Figure Eight's (Aspen) dataset that can be found at https://appen.com/datasets/combined-disaster-response-data. It is a dataset of thousands of messages (more than 26 000) collected during natural disasters from various sources.
For each message there are 36 possible categories (like 'medical_help', 'weather_related' and etc.)

# Project structure 
TBD

# Installation
1. In order to install the code and deploy the app locally please download from Github: `git clone https://github.com/alibekU/disaster_messages_categorization.git`.
2. You may want to set up a new virtual environment: `python3 -m venv /path/to/new/virtual/environment` 
3. Then, use pip to install all the needed packages: `pip install -r requirements.txt`

# Usage
After downloading, go to the the 'disaster_messages_categorization' folder and:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Disaster_response.db models/classifier.pkl`

2. Go to the app/ directory and run the following command to launch your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Author 
- Alibek Utyubayev
- Link to Linkedin: https://www.linkedin.com/in/alibek-utyubayev-74402721/

# Credits
Udacity Data Scientist Nano-Degree for the project structure and starter code

# Requirements
Listed in requirements.txt file:
- click==7.1.2
- Flask==1.1.2
- itsdangerous==1.1.0
- Jinja2==2.11.2
- joblib==0.17.0
- MarkupSafe==1.1.1
- nltk==3.5
- numpy==1.19.2
- pandas==1.1.3
- plotly==4.11.0
- python-dateutil==2.8.1
- pytz==2020.1
- regex==2020.10.11
- retrying==1.3.3
- scikit-learn==0.23.2
- scipy==1.5.2
- six==1.15.0
- SQLAlchemy==1.3.20
- threadpoolctl==2.1.0
- tqdm==4.50.2
- Werkzeug==1.0.1

