# -*- coding: utf-8 -*-

import pickle
import plac
import spacy
import jieba
import time
import os


@plac.annotations(
    model=("Model name. Defaults to blank 'zh_model'.", "option", "m", str),
    train_data=("Test data pickle file-name with path", "option", "t", str)
)
def main(model='zh_model', train_data=None):
    """Load the model, set up the pipeline and train the entity recognizer."""
    jieba.initialize()

    print('loading User dict...')
    jieba.load_userdict('dict.txt')
    if not os.path.exists('files'):
        os.mkdir('files')
    TRAIN_DATA = []
    with open(train_data, 'rb') as fr:
        TRAIN_DATA = pickle.load(fr)

    nlp = spacy.load("zh_model")
    count = 0
    with open('files/eval_result_' + str(time.strftime(
            '%Y%m%d_%H%M', time.localtime())) + '.txt', 'w+') as fw:
        for i, text in enumerate(TRAIN_DATA[1]):
            doc = nlp(text)
            l1 = [(ent.text, ent.label_) for ent in doc.ents]
            l2 = [(t[0], t[1]) for t in TRAIN_DATA[2][i]]
            if l1 != l2:
                print('Results: ', l1)
                print('Correct: ', l2, '\n----------\n')
                fw.write('Results: ' + str(l1) + '\n' +
                         'Correct: ' + str(l2) + '\n----------\n')
            count += len(list(set(l1).intersection(set(l2))))
        fw.write('Precision: ' + str(count) + ' / ' + str(TRAIN_DATA[0]))
        print('Precision: ', count, ' / ', TRAIN_DATA[0])

if __name__ == '__main__':
    st = time.time()
    plac.call(main)
    ed = time.time()
    print(' Cost: %.3f' % (ed - st), ' secs...')
