import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as Go
import numpy as np
import joblib
from sqlalchemy import create_engine
from support_functions import tokenize, cramers_v, create_heatmap_data


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/Disaster_response.db')
df = pd.read_sql_table('Messages', engine)
engine.dispose()

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # get the labels - Y dataframe - for visuals
    Y = df[df.columns[4:]]
    # get the category names
    category_names = Y.columns.to_list()
    # sum the messages for each category and sort 
    sums_categories = Y.sum(axis=0).sort_values()
    # get the names of the categories in the sorted order
    category_names_sorted = list(sums_categories.index)
    
    # create a matrix of correlations btw categories for heatmap graph
    corr = create_heatmap_data(Y, cramers_v)
    # create a dataframe from the matrix
    corr_df = pd.DataFrame(data=corr, index=category_names, columns=category_names)
   
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # Class distribution
        {
            'data': [
                Go.Bar(
                    x=sums_categories,
                    y=category_names_sorted,
                    orientation='h',
                    marker=dict(color = sums_categories,colorscale='YlGnBu')
                )        
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'height':800,
                'yaxis': {
                    'title': "Categories",
                    'automargin': True
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        },
        # Correlation between classes
        {
            'data':[
                Go.Heatmap(
                    x=category_names,
                    y=category_names,
                    z=corr_df,
                    colorscale='YlGnBu'
                )
            ],
            
            'layout': {
                'title': "Correlation between categories",
                'height':800,
                'yaxis': {
                    'automargin': True
                },
                'xaxis': {
                    'automargin': True
                }
            }
        },
        # Genre distribution
        {
            'data': [
                Go.Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color='yellowgreen')
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