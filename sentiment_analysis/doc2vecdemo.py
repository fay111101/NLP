#!/usr/bin/env python3
from gensim.test.utils import common_texts,get_tmpfile
from gensim.models import Word2Vec
path=get_tmpfile('word2vec.model')
model=Word2Vec(common_texts,size=100,window=5,min_count=1,workers=4)
model.save('word2vec.model')

model=Word2Vec.load('word2vec.model')
model.train([['hello','world']],total_examples=1,epochs=1)
vector=model.wv['computer']

from gensim.models import KeyedVectors
path=get_tmpfile('wordvectors.kv')
model.wv.save(path)
wv=KeyedVectors.load(path,mmap='r')
vector=wv['computer']
print(vector)

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
documents=[TaggedDocument(doc,[i]) for i,doc in enumerate(common_texts)]
model=Doc2Vec(documents,vector_size=5,window=2,min_count=1,workers=4)
from gensim.test.utils import get_tmpfile
fname=get_tmpfile('my_doc2vec_model')
model.save(fname)
model=Doc2Vec.load(fname)
#
model.delete_temporary_training_data(keep_doctags_vectors=True,keep_inference=True)
vector=model.infer_vector(['system','response'])
print('===================')
print(vector)

import gensim

TaggedDocument=gensim.models.doc2vec.TaggedDocument


