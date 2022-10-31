#!/usr/bin/env python
# coding: utf-8

# # `BERCHMANS KEVIN S`

# # `Sentiment Analysis on Movie Reviews`

# ###  Exercise - 1

# In[1]:


import pandas as pd


# In[2]:


file = pd.read_csv('train.tsv', sep='\t')


# In[3]:


file


# In[4]:


file.head()


# In[5]:


file.shape


# In[6]:


file.size


# In[7]:


file.describe()


# In[8]:


file.columns


# In[9]:


file.tail()


# In[10]:


file['Sentiment'].value_counts()


# ###  Exercise - 2

# In[11]:


a1 = file.loc[file.Sentiment == 0]
b1 = file.loc[file.Sentiment == 1]
c1 = file.loc[file.Sentiment == 2]
d1 = file.loc[file.Sentiment == 3]
e1 = file.loc[file.Sentiment == 4]


# In[12]:


small_rotten_train = pd.concat([a1[:200], b1[:200], c1[:200], d1[:200], e1[:200]])


# ### Exercise - 3

# In[13]:


small_rotten_train


# In[14]:


X = small_rotten_train.Phrase
X


# In[15]:


y = small_rotten_train.Sentiment
y


# In[16]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[17]:


from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')


# In[18]:


Lemmatizer = WordNetLemmatizer()


# In[19]:


def clean_reviews(review):
    tokens = review.lower().split()
    filtered_tokens = [
        Lemmatizer.lemmatize(w) for w in tokens if w not in stop_words
    ]
    return" ".join(filtered_tokens)


# In[20]:


temp = X.tolist()
fax = []


# In[21]:


for i in temp:
    fax.append(clean_reviews(i))    
n_X = pd.Series(fax)


# In[22]:


from sklearn.model_selection import train_test_split
X_train, y_train, X_test, y_test = train_test_split(n_X, y, test_size=0.20)


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(min_df=3, max_features=None,
                    ngram_range=(1,2), use_idf=1)
tf


# In[24]:


tf.fit_transform(X_train)


# In[25]:


from sklearn.naive_bayes import MultinomialNB


# In[26]:


clf = MultinomialNB()


# In[27]:


X_train


# In[28]:


# clf.fit(X_train, y_train)


# In[29]:


y_train


# In[ ]:




