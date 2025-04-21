#we are creating a model which tells whether a message is spam or real

#importing required modules:
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#laoding the dataset:
data = pd.read_csv('csv_files/mail_data.csv')
data.head(10)
#output will be:
#    Category	              Message
# 0	   ham	     Go until jurong point, crazy.. Available only ...
# 1	   ham	     Ok lar... Joking wif u oni...
# 2	   spam	     Free entry in 2 a wkly comp to win FA Cup fina...
# 3	   ham	     U dun say so early hor... U c already then say...
# 4	   ham	     Nah I don't think he goes to usf, he lives aro...
# 5	   spam	     FreeMsg Hey there darling it's been 3 week's n...
# 6	   ham	     Even my brother is not like to speak with me. ...
# 7	   ham       As per your request 'Melle Melle (Oru Minnamin...
# 8	   spam	     WINNER!! As a valued network customer you have...
# 9	   spam	     Had your mobile 11 months or more? U R entitle...

#now we will the size of our dataset:
data.shape
# output:  (5572, 2)

#so we will encode our target because computer understand only alphabetical and it has only 2 values:

data['Category'] = data['Category'].map({'ham': 1, 'spam': 0})
#this will do:
# denote :
# ham(real_message): represent as 1
# spam (fake) : represent as 0

#now we will if there are missing value is present or not in our dataset:
data.isnull().sum()
#output : 0,0... 
# so no null value is present

# now we will seperate our dataset:

X = data['Message']
y= data['Category']

#now we will split our data into training and testing set:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 2)

#now we will encode our feature:
# with help of TfidfVectorizerr:

vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

#now we call our model:
model = LogisticRegression()
model.fit(X_train_features, y_train)

#now we will predict:

pred_on_X_train = model.predict(X_test_features)
acc_on_X_train = accuracy_score(pred_on_X_train, y_train)
print(acc_on_X_train)
#output is : 0.9685887368184878
#which is very Good

#now we will create our predictive user_input

user_input = ()
converted = vectorizer.transform([user_input])
prediction = model.predict(converted)
print(prediction)
if prediction[0] == 1: 
  print('Real Message')
else:
  print('Spam Message')