#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
#   $ ./lda.py -c 0:20 -k 10 --alpha=0.5 --beta=0.5 -i 50

import numpy as np
#import javarandom as jr
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )


modelfile = "./model/final" # 模型文件的存储位置

class LDA:
    def __init__(self, K, alpha, beta, docs, V, smartinit=True):
        """    smartinit 初始化时使用Eq79为词t分配一个初始topic
        而普通init (else) 初始化时直接用均匀分布逼近多项分布  """
    
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs
        self.V = V
        self.M = len(self.docs)
        
        self.z_m_n = [] # topics of words of documents
        self.n_m_k = np.zeros((self.M, K)) + alpha     # word count of each document and topic
        self.n_k_t = np.zeros((K, V)) + beta # word count of each topic and vocabulary
        self.n_k = np.zeros(K) + V * beta    # word count of each topic
        self.n_m = np.zeros(self.M) + K * alpha

        self.theta = np.zeros((self.M, K))
        self.phi = np.zeros((K, V))

        self.N = 0
        for m, doc in enumerate(docs):
            self.N += len(doc)# lenth of all words in documents
            z_n = []
            for t in doc:
                if smartinit:
                    p_z = self.n_k_t[:, t] * self.n_m_k[m] / self.n_k #Eq79
                    z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                else:
                    z = np.random.randint(0, K)
                    #z = jr.Random(1).nextInt(K)                     
                z_n.append(z)
                #add newly estimated z to count variables
                self.n_m_k[m, z] += 1  
                self.n_k_t[z, t] += 1
                self.n_k[z] += 1
                self.n_m[m] += 1  #??
            self.z_m_n.append(np.array(z_n)) 
        #print self.z_m_n

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                z = self.z_m_n[m][n]   #z为第m篇文档第n个词的topic
                self.n_m_k[m, z] -= 1
                self.n_k_t[z, t] -= 1
                self.n_k[z] -= 1

                # sampling topic new_z for t 
                p_z = self.n_k_t[:, t] * self.n_m_k[m] / self.n_k   #Eq.79
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
#                p=[0] * self.K
#                for k in range(0, self.K):
#                    p[k] = self.n_k_t[:, t] * self.n_m_k[m] / self.n_k
#                for k in range(1, self.K):
#                    p[k] += p[k - 1]
#                new_z = jr.Random().nextDouble() * p[self.K - 1].argmax()
                
                # set z the new topic and increment counters
                self.z_m_n[m][n] = new_z                
                self.n_m_k[m, new_z] += 1
                self.n_k_t[new_z, t] += 1
                self.n_k[new_z] += 1
                
    def worddist(self):
        """get topic-word distribution"""
        return self.n_k_t / self.n_k[:, np.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            #theta = self.n_m_k[m] / (len(self.docs[m]) + Kalpha)
            self.theta[m] = self.n_m_k[m] / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= np.log(np.inner(phi[:,w], self.theta[m]))
            N += len(doc)
        #print self.theta
        return np.exp(log_per / N), self.theta

def lda_learning(lda, iteration, voca):
    pre_perp, theta = lda.perplexity()
    print "initial perplexity=%f" % pre_perp
    print "Sampling %d iterations" % iteration
    for i in range(iteration):
        print 'Iteration %d ...' % (i+1)
        lda.inference()
        perp, t = lda.perplexity()  # t此处无意义
        print "p=%f" %perp
        if pre_perp:
            if pre_perp < perp:
                output_word_topic_dist(lda, voca)
                pre_perp = None
            else:
                pre_perp = perp
    print 'End sampling.'
    phi = output_word_topic_dist(lda, voca)
    save_model(lda, theta, phi)

def output_word_topic_dist(lda, voca):
    zcount = np.zeros(lda.K, dtype=int)
    wordcount = [dict() for k in xrange(lda.K)]
    for xlist, zlist in zip(lda.docs, lda.z_m_n):
        for x, z in zip(xlist, zlist):
            zcount[z] += 1
            if x in wordcount[z]:
                wordcount[z][x] += 1
            else:
                wordcount[z][x] = 1

    phi = lda.worddist()
    #print phi
    for k in xrange(lda.K):
        print "\n-- topic: %d (%d words)" % (k, zcount[k])
        for w in np.argsort(-phi[k])[:20]:
            print "%s: %f (%d)" % (voca[w], phi[k,w], wordcount[k].get(w,0))
    
    with open(modelfile+'.twords', 'w') as ftwords:   #save topic-word 
        for k in xrange(lda.K):
            ftwords.write("topic: %d (%d words)" % (k, zcount[k]))
            ftwords.write('\n')
            for w in np.argsort(-phi[k])[:20]:
                ftwords.write("\t %s: %f (%d)" % (voca[w], phi[k,w], wordcount[k].get(w,0))) 
                ftwords.write('\n')
       
    return phi

def save_model(lda, theta, phi):#<yt>save theta and phi
    with open(modelfile+'.theta', 'w') as ftheta:#存放theta
        for x in xrange(lda.M):
            for y in xrange(lda.K):
                ftheta.write(str(theta[x,y]) + ' ')
            ftheta.write('\n')
    with open(modelfile+'.phi', 'w') as fphi:#存放phi
        for x in xrange(lda.K):
            for y in xrange(lda.V):
                fphi.write(str(phi[x, y]) + ' ')
            fphi.write('\n')
               

def main_eng():
    import optparse
    import vocabulary
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename")
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("-s", dest="smartinit", action="store_true", help="smart initialize of parameters", default=False)
    parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true", default=False)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")

    if options.filename:
        corpus = vocabulary.load_file(options.filename)
    else:
        corpus = vocabulary.load_corpus(options.corpus)
        if not corpus: parser.error("corpus range(-c) forms 'start:end'")
    if options.seed != None:
        np.random.seed(options.seed)

    voca = vocabulary.Vocabulary(options.stopwords)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

    lda = LDA(options.K, options.alpha, options.beta, docs, voca.size(), options.smartinit)
    print "corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.K, options.alpha, options.beta)

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    lda_learning(lda, options.iteration, voca)

def main_chi():
    import optparse
    import vocabulary_chi
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="parent_dir", help="parent_dir")
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    parser.add_option("-s", dest="smartinit", action="store_true", help="smart initialize of parameters", default=False)
    parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true", default=False)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.parent_dir or options.corpus): parser.error("need corpus parent_dir(-f) or corpus range(-c)")

    if options.parent_dir:
        corpus = vocabulary_chi.load_file(options.parent_dir)
    else:
        corpus = vocabulary_chi.load_corpus(options.corpus)
        if not corpus: parser.error("corpus range(-c) forms 'start:end'")
    if options.seed != None:
        np.random.seed(options.seed)

    voca = vocabulary_chi.Vocabulary(options.stopwords)
    
    with open('word.txt', 'w') as wd:
        for w in xrange(len(voca.vocas)):
            wd.write(voca.vocas[w])
        wd.write('\n')
    
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

    lda = LDA(options.K, options.alpha, options.beta, docs, voca.size(), options.smartinit)
    print "corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.K, options.alpha, options.beta)

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    lda_learning(lda, options.iteration, voca)
    
if __name__ == "__main__":
    #main_eng()
    main_chi()
