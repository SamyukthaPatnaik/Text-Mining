#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import spacy

from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS


# In[9]:


Elon=pd.read_csv("C:\\Users\\ASUS\\Downloads\\DATA SCIENCE\\ASSIGNMENTS\\Text Mining\\Elon_musk.csv",encoding = "ISO-8859-1")


# In[10]:


Elon.head()


# In[11]:


Elon.drop(['Unnamed: 0'],inplace=True,axis=1)
Elon


# In[12]:


Elon = [Text.strip() for Text in Elon['Text']] 


# In[15]:


Elon=[Text  for Text in Elon if Text]
Elon


# In[16]:


# joining the list of comments into a single text/string

text = ' '.join(Elon)


# In[17]:


text


# In[18]:




len(text)


# In[19]:


no_punc_text = text.translate(str.maketrans('','',string.punctuation))


# In[20]:


no_punc_text


# In[21]:


import re
no_url_text=re.sub(r'http\S+', '', no_punc_text)
no_url_text


# In[22]:


import nltk
from nltk.tokenize import word_tokenize  #Tokenization
nltk.download('punkt')
text_tokens=word_tokenize(no_url_text)
print(text_tokens)


# In[23]:


nltk.download('stopwords')


# In[24]:




len(text_tokens)


# In[26]:


# Remove Stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')

sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# In[27]:


# Normalize the data
lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])


# In[28]:


# Stemming (Optional)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])


# In[29]:


# Lemmatization
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)


# In[30]:


lemmas = [token.lemma_ for token in doc]


# In[31]:


clean_tweets=' '.join(lemmas)
clean_tweets


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)


# In[33]:


print(cv.vocabulary_)


# In[34]:


print(cv.get_feature_names()[100:200])


# In[35]:




print(tweetscv.toarray()[100:200])


# In[36]:


print(tweetscv.toarray().shape)


# In[37]:


cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)  #Bigrams and Trigrams
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)


# In[38]:


print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# In[39]:


from sklearn.feature_extraction.text import TfidfVectorizer    #Tf idf Vectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


# In[40]:


print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matix_ngram.toarray())


# In[41]:


def plot_cloud(wordcloud):             #word cloud
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')
    
# Generate Word Cloud

STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_tweets)
plot_cloud(wordcloud)


# In[42]:


# Parts Of Speech (POS) Tagging
nlp=spacy.load('en_core_web_sm')

one_block=clean_tweets
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[43]:


for token in doc_block[100:200]:
    print(token,token.pos_)


# In[44]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# In[45]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# In[46]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs',color = 'blue');


# In[49]:


afinn = pd.read_csv(r"C:\\Users\\ASUS\\Downloads\\DATA SCIENCE\\ASSIGNMENTS\\Text Mining\\Afinn.csv",sep=",", encoding='cp1252', 
                 error_bad_lines=False,warn_bad_lines=False)
afinn.head()


# In[53]:


affinity_scores = afinn.set_index('word')['value'].to_dict()
affinity_scores


# In[54]:


# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon = affinity_scores


def calculate_sentiment(text:str=None):
    sent_score = 0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[55]:


# manual testing
calculate_sentiment(text='great')


# In[56]:


from nltk.tokenize import sent_tokenize


# In[58]:


sentence = nltk.tokenize.sent_tokenize(' '.join(Elon))
sentence[5:15]


# In[59]:


sent_df = pd.DataFrame(sentence, columns = ['sentences'])
sent_df


# In[60]:


# Calculating sentiment value for each sentences
sent_df['sentiment_value']=sent_df['sentences'].apply(calculate_sentiment)
sent_df['sentiment_value']


# In[61]:


# how many words are there in a sentences?
sent_df['word_count']=sent_df['sentences'].str.split().apply(len)
sent_df['word_count']


# In[62]:


sent_df.sort_values(by='sentiment_value')


# In[63]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[64]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[65]:


# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]


# In[66]:


# Adding index cloumn
sent_df['index']=range(0,len(sent_df))
sent_df


# In[67]:


import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])


# In[68]:


plt.figure(figsize=(15, 10))
sns.lineplot(x = sent_df.index, y = sent_df['sentiment_value'], data = sent_df)






