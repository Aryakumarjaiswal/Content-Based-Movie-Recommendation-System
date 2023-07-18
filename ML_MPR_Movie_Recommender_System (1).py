#!/usr/bin/env python
# coding: utf-8

# In[2]:


#ML MPR
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ast


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')     #Loading movie.csv file
credits = pd.read_csv('tmdb_5000_credits.csv')       #Loading credit.csv file


# In[14]:


movies.head()
#movies.head(1)['keywords'].values


# In[15]:


#credits.head(1)
credits.shape


# In[ ]:


credits.head(1)['cast'].values


# In[ ]:


#Since We have two dataframes so in order to make things simple
#we will do merging it can be on any common column
movies.shape


# In[ ]:


credits.shape


# In[ ]:


movies=movies.merge(credits, on='title')#On the basis of title


# In[2]:


movies.info()


# In[3]:


movies = movies[['movie_id','title','overview','genres','cast','crew','keywords']]


# In[4]:


#movies.head(2)
movies.loc[0,'overview']


# In[5]:


movies.isnull()


# In[6]:


movies.isnull().sum()#Finding missing data


# In[14]:


movies.dropna()


# In[15]:


movies.isnull().sum()


# In[16]:


movies = movies.drop_duplicates()


# In[17]:


movies.loc[:,'title']


# In[18]:


movies.loc[:,'genres']#Genres Formatting


# In[19]:


#movies.loc[movies.title=='Avatar','id']
#movies.loc[:,'budget':'original_title']
#movies.loc[movies.budget>=100000000,'title']


# In[20]:


movies.loc[0,'genres']


# In[21]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
        


# In[22]:



movies['genres']=movies['genres'].apply(convert)
movies.loc[0,'genres']


# In[23]:


movies['keywords']=movies['keywords'].apply(convert)
#movies.loc[0,'keywords']


# In[24]:


movies.head()


# In[25]:


#Cast Function:

def toCast(obj):
    cnt=0
    L=[]
    for i in obj:
        if (cnt==3):
            break;
        else:
            L.append(i['name'])
            cnt+=1 
    return L
        


# In[26]:


movies['cast']=movies['cast'].apply(toCast)
movies.loc[0,'cast']


# In[27]:


#movies.head(3)
movies['genres']


# In[28]:


#Director
def Directr(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if(i['job']=='Director'):
            L.append(i['name'])
            return L
        


# In[29]:


movies['crew']=movies['crew'].apply(Directr)
#movies.loc[0,'genres']
#movies.loc[0,'cast']
#movies.loc[0,'crew']


# In[30]:


movies['overview']=movies['overview'].apply(lambda x:x.split())

#print(movies[movies['overview'].apply(lambda x: isinstance(x, float))])
movies.loc[0,'overview']


# In[31]:


movies['overview'] = movies['overview']


# In[32]:


movies.head()


# In[33]:


#def collapse(L):
 #   L1 = []
  #  for i in L:
   #     L1.append(i.replace(' ',""))
    #return L1


# In[34]:


movies['cast'] = movies['cast'].apply(lambda x:[i.replace(' ',"")  for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(' ',"")  for i in x])
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(' ',"")  for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(' ',"")  for i in x])


# In[ ]:


movies.head()


# In[ ]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new=movies[['title','tags','movie_id']]

new.head()


# In[ ]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x) if isinstance(x, (list, tuple)) else x)


# In[ ]:


#new.loc[0,'tags']
new['tags']=new['tags'].str.lower()


# In[ ]:


new['tags'].isna().sum()
new['tags'].fillna('', inplace=True)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
    


# In[35]:


vector = cv.fit_transform(new['tags']).toarray()#[]=>arr


# In[36]:


vector


# In[ ]:





# In[37]:


import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[38]:


def stem(text):
    y=[]
    for i in text.split():
        ps.stem(i)


# In[39]:


new['tags']=new['tags'].apply(stem)


# In[40]:


similarity = cosine_similarity(vector)


# In[41]:


similarity


# In[42]:


def recommend(movie):
    movie_index = new[new['title'] == movie].index[0]
    distances = similarity[movie_index] 
    mlist=sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:5]
    for i in mlist:
        print(new.iloc[i[0]].title)


# In[43]:


recommend('Avatar')


# In[44]:


import pickle
pickle.dump(new,open('movies.pkl','wb'))


# In[ ]:





# In[45]:


pickle.dump(new.to_dict(),open('movie_dict.pkl','wb'))


# In[47]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:




