from os import read
from flask import Flask, request, render_template

import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from dashboard import getvaluecounts, getlevelcount, getsubjectsperlevel, yearwiseprofit

app = Flask(__name__,template_folder='templates')

def cosinesimmat(cvmat):

    return cosine_similarity(cvmat)

def getcosinemat(df):

    countvect = CountVectorizer()
    cvmat = countvect.fit_transform(df['Clean_title'])
    return cvmat

def readdata():

    df = pd.read_csv('UdemyCleanedTitle.csv')
    return df

def getcleantitle(df):

    df['Clean_title'] = df['course_title'].apply(nfx.remove_stopwords)
    df['Clean_title'] = df['Clean_title'].apply(nfx.remove_special_characters)
    return df

def extractfeatures(recdf):

    course_url = list(recdf['url'])
    course_title = list(recdf['course_title'])
    course_price = list(recdf['price'])

    return course_url, course_title, course_price

def searchterm(term, df):
    resultdf = df[df['course_title'].str.contains(term)]
    top6 = resultdf.sort_values(by='num_subscribers', ascending=False).head(6)
    return top6

def recommend_course(df, title, cosine_mat, num_rec):

    course_index = pd.Series(
    df.index,index=df['course_title']).drop_duplicates()

    index = course_index[title]

    scores = list(enumerate(cosine_mat[index]))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    selected_course_index = [i[0] for i in sorted_scores[1:]]

    selected_course_score = [i[1] for i in sorted_scores[1:]]

    rec_df = df.iloc[selected_course_index]

    rec_df['similarity_score'] = selected_course_score

    final_recommended_courses = rec_df[['course_title','similarity_score','url','price','num_subscribers']]

    return final_recommended_courses.head(num_rec)

@app.route('/', methods=['GET','POST'])
def hello_world():
    
    if request.method == 'POST':

        my_dict = request.form
        title = my_dict['course']
        print(title)
        try:
            df = readdata()
            df = getcleantitle(df)
            cvmat = getcosinemat(df)

            num_rec = 6
            cosine_mat = cosinesimmat(cvmat)

            recdf = recommend_course(df, title, cosine_mat, num_rec)

            course_url, course_title, course_price = extractfeatures(recdf)

            dictmap = dict(zip(course_title, course_url))

            if len(dictmap) != 0:

                return render_template('index.html', coursemap=dictmap,
                                       coursename=title, showTitle=True)
            else:
                return render_template('index.html', showerror=True, coursename=title)
            
        except:

            resultdf = searchterm(title, df)
            if resultdf.shape[0] > 6:
                resultdf = resultdf.head(6)

                course_url, course_title, course_price = extractfeatures(resultdf)
                coursemap = dict(zip(course_title, course_url))

                if len(coursemap) != 0:
                    return render_template('index.html', coursemap=coursemap,
                                       coursename=title, showTitle=True)

                else:
                    return render_template('index.html', showerror=True, coursename=title)
                
            else:
                course_url, course_title, course_price = extractfeatures(resultdf)
                coursemap = dict(zip(course_title, course_url))

                if len(coursemap) != 0:
                    return render_template('index.html', coursemap=coursemap,
                                       coursename=title, showTitle=True)

                else:
                    return render_template('index.html', showerror=True, coursename=title)
                
    return render_template('index.html')

@app.route('/dashboard', methods=['GET','POST'])
def dashboard():
    
    df = readdata()
    valuecounts = getvaluecounts(df)

    levelcounts = getlevelcount(df)

    subjectperlevel = getsubjectsperlevel(df)

    yearwiseprofitmap, subscribercountmap, profitmonthwise, monthwisesub = yearwiseprofit(df)

    return render_template('dashboard.html',valuecounts=valuecounts, levelcounts=levelcounts, subjectperlevel=subjectperlevel,
                           yearwiseprofitmap=yearwiseprofitmap, subscribercountmap=subscribercountmap, profitmonthwise=profitmonthwise,
                           monthwisesub=monthwisesub)

if __name__ == '__main__':
    app.run(debug=True)