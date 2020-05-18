import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
import model
new_movies=pd.read_csv('new_movies.csv')
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    try:
        movie = request.args.get('movie')  
        r=model.recommendation(movie)
        #movie = movie.upper()
        if type(r)==type('string'):
            return render_template('recommend.html',movie=movie,r=r,t='s')
        else:
            return render_template('recommend.html',movie=movie,r=r,t='l')
    except Exception as e:
        return "Movie not in database",200
        print(e)


if __name__ == '__main__':
    app.run()
