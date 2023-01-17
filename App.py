# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:42:39 2022

@author: shaheheer

"""

import streamlit as st
import pandas as p
import numpy as n
import nltk
import sklearn


nltk.download('punkt')
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.stem import WordNetLemmatizer
import re
import string
import pickle

#import models

tfidf = pickle.load(open("tf_vectorizer.pkl", "rb"))
lda_model = pickle.load(open("lda_model.pkl", "rb"))
neg_stopwords = pickle.load(open("stop_words2.pkl","rb"))
cv = pickle.load(open("cv_vectorizer.pkl","rb"))
model = pickle.load(open("XtraTree.pkl","rb"))

l = WordNetLemmatizer()

# Lemmatization
def preprocessor(text):
    x = text.lower()
    x = re.sub("\d+[/?]\w+[/?]\w+:|\d+[|]\w+[|]\w+:|\d+[/]\w+[/]\w+[(]\w+[)]:?", "", x)
    x = re.sub("int[a-z]+d$", "interested", x)
    x = re.sub("[\d+-?,'.]", "", x)
    x = [i for i in nltk.word_tokenize(x) if i not in neg_stopwords and len(i)>1 and i not in string.punctuation] 
    x = [l.lemmatize(i) for i in x]
    return " ".join(x)

#Topic Modelling System

def topic_model(user):
    a= user
    x = preprocessor(a)
    x = cv.transform([x])
    lda_x = lda_model.transform(x)
    tpic = []
    tpc = lambda x : "not interested" if x == 0 else "interested"
    for i,topic in enumerate(lda_x[0]):
        print("Topic ",i,": ",topic*100,"%")
        tpc_name = tpc(i)
        prc = topic*100
        tpic.append([tpc_name,prc])
    return tpic

# Classification Model
def Status(user):
    x = user
    x = preprocessor(x)
    x = tfidf.transform([x])
    x = model.predict(x.toarray())
    if x == 1:
        return "Not Convertable"
    else:
        return "Convertable"

def main():
    st.subheader("Customer Relationship Management")
    st.write("Adding Intelligence to CRM and Improving Conversion Ratio")
    user = st.text_area("Enter the description of the conversation here : ")  
    col1, col2 = st.columns([1,1])    
    
    with col1:
        if st.button("Topic Modelling"):
            topics = topic_model(user)
            df1 = p.DataFrame(topics[0])
            df2 = p.DataFrame(topics[1])
            st.write(df1.to_string(header=None, index=False))
            st.write(df2.to_string(header=None, index=False))
                  
    with col2:
        if st.button("Classification"):
            result = Status(user)
            if result == "Convertable":
                st.success(result)
            else:
                st.error(result)

if __name__ == "__main__":
    main()


