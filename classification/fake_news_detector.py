import numpy as np
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


#importing dataset:
true_news = pd.read_csv('truenews.csv')
fake_news = pd.read_csv('fakenews.csv')

true_news['label'] = 1
fake_news['label'] = 0

df_merge = pd.concat([true_news, fake_news], axis= 0)

#it will merge 2 dataset into 1

#now we will drop those feautures which are not required

df = df_merge.drop(['title', 'date', 'subjects'])


#now we will if there is no empty value:
df.isnull().sum()
#result is 0.
#so there are not missing values so we will move forward

#now we will shufffle the dataset for better training

df = df.sample(frac=1)  #frac 1 means it will shuffle whole dataset
df.reset_index(inplace= True) #reset_index : it will reset the index as it previously shuffled
#inplace true means  changes are made directly to original data set
df.drop(['index'],axis=1,  inplace= True)


#now we will aplly a fn to simplify our traning model

def simplify(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)  #bracketed text
    text = re.sub("\\W"," ",text)# - Non-word characters
    text = re.sub('https?://\S+|www\.\S+', '', text) # - URLs
    text = re.sub('<.*?>+', '', text) # - HTML tag
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  #Punctuation
    text = re.sub('\n', '', text) # - Newlines
    text = re.sub('\w*\d\w*', '', text) # - Words containing number
    return text
#it will remove all of this from our datset
#in this process it will remove all of these items for good training
# items are:  Bracketed text

df['text'] = df['text'].apply(simplify)

#now we will decleare our features and target

x = df['text']
y = df['label']

# now we will send these for training data:
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify= y, test_size= 0.25, random_state=2)

#now we will call a module which converts text data into numeric data
#bcz computers only understand numeric data

converter = TfidfVectorizer()
x_train = converter.fit_transform(x_train)
x_test = converter.transform(x_test)

#now we call a model for classification problem
#here we are calling LogisticRegression
 
model = LogisticRegression()
model = model.fit(x_train, y_train)

prediction_x_test = model.predict(x_test)

#we will check the score of our model



print(classification_report(y_test, prediction_x_test)) #it willl give classification report of our model

# now lets create a prdictive system
input_your_message = []
input_your_message  = converter.transform(input_your_message)
prediction = model.predict(input_your_message)
print(prediction)
if prediction[0] == 0:
    print('msg is fake')
else:
    print('Message is real')