# -*- coding: utf-8 -*-
from gensim.corpora import WikiCorpus
'''
    extract data from wiki dumps(*articles.xml.bz2) by gensim.
'''

if __name__ == '__main__':
    inp = 'zhwiki-20181220-pages-articles-multistream.xml.bz2'
    outp = 'zhwiki_latest.txt'
    i = 0

    with open(outp, 'w') as output:
        wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
        for text in wiki.get_texts():
            output.write(" ".join(text) + "\n")
