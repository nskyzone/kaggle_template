
# coding: utf-8

# # 处理数据
# 
# NLP问题,去停用词，分词，词向量

# In[15]:


import pandas  as pd
import numpy as np
import re
import jieba
import jieba.posseg
import jieba.analyse
import codecs


# # 分词 停用词

# In[ ]:


# 分词
def split_word(query, stopwords):
    wordList = jieba.cut(query)
    num = 0
    result = ''
    for word in wordList:
        word = word.rstrip()
        word = word.rstrip('"')
        if word not in stopwords:
            if num == 0:
                result = word
                num = 1
            else:
                result = result + ' ' + word
    return result.encode('utf-8')

# 处理停用词 Series
def deal_stop_word(data):
    stopwords = {}
    for line in codecs.open('../data/stop.txt','r','gbk'):
        stopwords[line.rstrip()]=1
    doc = data.map(lambda x:split_word(x,stopwords))
    return doc

def preprocess(data):
    pass


# # TF-IDF

# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
def get_tfidf(train,predict):
    vector = TfidfVectorizer(max_features=10000)
    train_words_len = train.words.shape[0]
    tfidf = vector.fit_transform(train.words.append(predict.words))
    tfidf_dt = tfidf.toarray()
    x_train = tfidf_dt[:train_words_len,:]
    x_predit = tfidf_dt[train_words_len:,:]
    return x_train,x_predict

def save_tfidf(train_tfidf, predict_tfidf):
    np.save('../data/train_tfidf',train_tfidf)
    np.save('../data/predict_tfidf',predict_tfidf)


# # embedding

# In[24]:


from gensim.models.word2vec import Word2Vec
def get_embedding(docs,model_path='embedding'):
    model = Word2Vec(docs.values.tolist(),
                     size=100,  # 词向量维度
                     min_count=5,  # 词频阈值
                     window=5)  # 窗口大小
    model.save('../data/' + model_path + '.mldel')


# 构建embedding_matrix    
def get_embedding_matrix(model_path,tok_raw,save=False):
    model = Word2Vec.load(model_path)
    word_index = tok_raw.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, model.vector_size))

    for word, i in word_index.items():
        if word in model.wv.vocab:
            vector = model.wv[word]
            if vector is not None:
                embedding_matrix[i] = vector
    if save:
        np.save('embedding_matrix',embedding_matrix)
    return embedding_matrix


# # 加载数据

# In[17]:


train_data = pd.read_csv('../data/train_first.csv')
predict_data = pd.read_csv('../data/predict_first.csv')
train.head()


# In[23]:


train_data['doc'] = deal_stop_word(train_data['Discuss'])
predict_data['doc'] = deal_stop_word(predict_data['Discuss'])
# 整合数据
docs = np.hstack([train_data.doc.values,predict_data.doc.values])


# In[11]:


b_train = pd.concat((a_train1,a_train2,train_3 ,train_4,train_5),ignore_index=True)
bb=b_train.drop(columns=['Id'])
bb.to_csv('../data/clean_train2.csv',index=None)

