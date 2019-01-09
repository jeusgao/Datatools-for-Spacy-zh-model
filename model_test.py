# -*- coding: utf-8 -*-

import spacy
import jieba

jieba.initialize()
print("loading User dict...")
jieba.load_userdict("dict.txt")

nlp = spacy.load("zh_model")
while True:
    qry = input("> ")
    doc = nlp(qry)
    words = [(ent.text, ent.label_) for ent in doc.ents]
    for word, flag in words:
        print(flag, word)
