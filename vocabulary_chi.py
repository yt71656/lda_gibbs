#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:18:14 2014

@author: lenovo
"""

#import jieba
import os
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
           
corpus = []
def load_file(parent_dir):#文本预处理（读入文档，分词）
    fname_list = os.listdir(parent_dir)
    for fname in fname_list:
        f = open(parent_dir + '\\' + fname,'r')
        raw = f.read().decode('utf-8')
        #word_list = list(jieba.cut(raw, cut_all = False))
        word_list = list(raw.split())
        corpus.append(word_list)
    f1 = open('wordlist.txt', 'w')
    for f in corpus:# output corpus after segment
        for w in f:
            #w = w.strip() 
            f1.write(w)
            f1.write(' ')
        f1.write('\n')
    return corpus

stopwords_list = open("stoplis.txt", "r").read()

def is_stopword(w):
    return w in stopwords_list

class Vocabulary:
    def __init__(self, excluds_stopwords=False):
        self.vocas = []        # id to word
        self.vocas_id = dict() # word to id
        self.docfreq = []      # id to document frequency
        self.excluds_stopwords = excluds_stopwords

    def term_to_id(self, term):
        if self.excluds_stopwords and is_stopword(term): return None
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
            self.docfreq.append(0)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def doc_to_ids(self, doc):
        #print ' '.join(doc)
        list = []
        words = dict()
        for term in doc:
            id = self.term_to_id(term)
            if id != None:
                list.append(id)
                if not words.has_key(id):
                    words[id] = 1
                    self.docfreq[id] += 1
        if "close" in dir(doc): doc.close()
        #print list       
        return list

    def cut_low_freq(self, corpus, threshold=1):#去掉词频<=1的
        new_vocas = []
        new_docfreq = []
        self.vocas_id = dict()
        conv_map = dict()
        for id, term in enumerate(self.vocas):
            freq = self.docfreq[id]
            if freq > threshold:
                new_id = len(new_vocas)
                self.vocas_id[term] = new_id
                new_vocas.append(term)
                new_docfreq.append(freq)
                conv_map[id] = new_id
        self.vocas = new_vocas
        self.docfreq = new_docfreq

        def conv(doc):
            new_doc = []
            for id in doc:
                if id in conv_map: new_doc.append(conv_map[id])
            return new_doc
        return [conv(doc) for doc in corpus]

    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)

    def is_stopword_id(self, id):
        return self.vocas[id] in stopwords_list

