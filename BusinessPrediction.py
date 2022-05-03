import warnings

import streamlit as st 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn import linear_model
sns.set()
from sklearn.datasets import load_boston
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

# Import the numpy and pandas package
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns


header = st.container()
dataset = st.container()
features =st.container()
model_training = st.container()


@st.cache
def get_data(filename):     
     df= pd.read_csv(filename)
     return df

"""
# Business Rating Prediction Model
""" 

"""
## Business improvization model
""" 

"""
### Dataset
""" 
import requests
url = 'https://raw.githubusercontent.com/soumilshah1995/NLP-Model-and-Data-Analysis-on-YELP-Dataset/master/yelp.csv'
res = requests.get(url, allow_redirects=True)
with open('yelp.csv','wb') as file:
     file.write(res.content)
df = get_data('yelp.csv')     
       
             
if st.checkbox("Show dataset"):
       st.dataframe(df.head(20))
"""
## Data Description
"""      
st.write(df.describe())
st.write(df.isnull().sum()*100/df.shape[0])
 
"""
## Heatmap of correlated values
"""     
st.text("Heatmap of correlated values")     
fig, ax = plt.subplots()

sns.heatmap(df.corr(),cmap="YlGnBu", annot = True)
st.write(fig)

"""
### Split the dataset
"""
left_column, right_column = st.columns(2)

# test size
test_size = left_column.number_input(
				'Testing-dataset size (rate: 0.0-1.0):',
				min_value=0.0,
				max_value=1.0,
				value=0.2,
				step=0.1,
				 )

# random_seed
random_seed = right_column.number_input('Set random seed (0-):',
							  value=0, step=1,
							  min_value=0)

# split the dataset
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,5), analyzer='char')
X = tfidf.fit_transform(df['text'])
Y = df['stars']
X_train, X_test, Y_train, Y_test = train_test_split(
	X,Y, 
	test_size=test_size, 
	random_state=random_seed
	)
clf =LinearSVC()
import numpy as np
from sklearn.linear_model import LinearRegression

# lr = linear_model.LinearRegression()
clf.fit(X_train, Y_train)

predicted = clf.predict(X_test)
st.text(classification_report(Y_test, predicted))

x= st.text_input('Enter review', 'Life of Brian')

vec= tfidf.transform([x])
st.write("Predicted Rating")
st.write(clf.predict(vec))




