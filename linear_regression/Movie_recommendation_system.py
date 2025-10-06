# #importing required modules
import pandas as pd
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# #loading the data set:
data = pd.read_csv('movies.csv')

data.shape
# (4803, 24)
# so we will take only relevent column which is required:

categorical =  ['genres', 'keywords', 'tagline', 'cast', 'director']

# we will fill missing values with null string:
for feature in categorical:
    data[feature] = data[feature].fillna('')
    
    #now there are no missing values

#now we will combine our data
combined = data['genres']+ ' ' + data['keywords']+ ' ' + data['tagline']+ ' '+ data['cost']+ ' ' + data['director']

#now we will vectorize those data using tfidfvectorizer:
# tf means term frequency, idf means inverse document frequency
# tf-idf is a numerical statistic that is intended to reflect how important a word is to
vectorizer = TfidfVectorizer()
# .fit_transform will learn the vocabulary and idf from training set, and return term-document matrix.
feature = vectorizer.fit_transform(combined) 

#now we will check the symmetric score using:
# cosine_similarity:
#cosine similarity is a metric used to measure how similar the documents are irrespective of their size.
similar = cosine_similarity(feature)

#now we will create a predictive system:

user_input = input('Enter the moive name: ')
#finding the title:
title = difflib.get_close_matches(user_input, data['title'])

cloesest_value = title[0]

#getting the index:
index = data[data.title == cloesest_value]['index'].values[0]

#getting the similarity
similarity = list(enumerate(similar[index]))

soritng =  sorted(similarity, key= lambda x:x[1], reverse= True)

for i in range(0,11):
    index = soritng[i][0]
    title =data[data.index == index]['title'].values[0]
    print('suggestions are:')
    print(i +1, title)
    