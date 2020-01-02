# -*- coding: utf-8 -*-
import random
import time
from functools import reduce
import plac
import pickle
import json
import os
# from numba import jit, njit


def loadvocabs(src_dir):
    d_slots = None
    d_sl = None
    fname = os.path.join(src_dir, 'slots.json')
    with open(fname, 'r') as fr:
        d_slots = json.loads(fr.read())[0]
    vocabs = '¥¥¥'.join(
        ['¥¥¥'.join(a[1]) for a in d_slots.items()]).split('¥¥¥')

    fname = os.path.join(src_dir, 'sl_dict.json')
    with open(fname, 'r') as fr:
        d_sl = json.loads(fr.read())[0]

    with open('dict.txt', 'r') as fr:
        ls_dic = fr.readlines()
    wds_dic = [ld.split(' ')[0] for ld in ls_dic]
    vocabs = list(set(vocabs).difference(set(wds_dic)))
    print('New words save into Jieba Dict: \n', vocabs)

    with open('dict.txt', 'a+') as fw:
        fw.writelines(v + ' 100000\n' for v in vocabs)
    return d_slots, d_sl


def _getlines(lines):
    tmp_ls = []
    lines = lines.split(',')
    for l in lines:
        if '-' in l:
            for i in range(int(l.split('-')[0]), int(l.split('-')[1])):
                tmp_ls.append(i)
        else:
            tmp_ls.append(int(l))
    return tmp_ls


def _get_corpus(tpl, d_slots, d_sl):
    print('TEMPLATE: ', tpl)
    ls = []
    tpl = tpl.split('@@')
    for t in tpl:
        ls.append(d_sl.get(t, []) if 'SPL__' in t else d_slots.get(t, []))

    total = reduce(lambda x, y: x * y, map(len, ls))
    cs = []
    for i in range(0, total):
        step = total
        tempItem = []
        for l in ls:
            step /= len(l)
            tempItem.append(l[int(i / step % len(l))])
        cs.append(tuple(tempItem))
    yield cs


def _gen_traindata(tpl, cs, d_slots, d_sl):

    TRAIN_DATA = []
    for vs in cs:
        text = ''
        d_entities = {}
        d_entities['entities'] = []
        for i, v in enumerate(vs):
            text += v
            if 'SPL__' in tpl.split('@@')[i]:
                continue
            d_entities['entities'].append(
                tuple((text.find(v), text.find(v) + len(v),
                       tpl.split('@@')[i])))
        TRAIN_DATA.append(tuple((text, d_entities)))
    yield TRAIN_DATA


def gener(template, lines, d_slots, d_sl):
    print('Generating training data set...')
    TRAIN_DATA = []
    input_lines = []

    with open(template, 'r') as fr:
        tpls = fr.readlines()
    if lines:
        input_lines = _getlines(lines)
    else:
        for i in range(len(tpls)):
            input_lines.append(i)

    for i in input_lines:
        cs = next(_get_corpus(tpls[i].replace('\n', ''), d_slots, d_sl))
        TRAIN_DATA += next(_gen_traindata(
            tpls[i].replace('\n', ''), cs, d_slots, d_sl))
    yield TRAIN_DATA


@plac.annotations(
    template=("Data template file name.", "option", "t", str),
    lines=(
        "Lines read from data template file, lines range connected with '-', \
        multi lines or ranges seperate with ',' default with all lines, \
        donnot fill whitespace in... ", "option", "l", str),
    output_dir=("Optional output directory", "option", "o", str),
    src_dir=("Resources directory, default in folder 'ref'",
             "option", "r", str)
)
def main(template=None, output_dir='model', lines=None, src_dir='ref'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    d_slots, d_sl = loadvocabs(src_dir)

    TRAIN_DATA = next(gener(template, lines, d_slots, d_sl))
    print(len(TRAIN_DATA))

    fname = os.path.join(output_dir, 'train_data_' + str(time.strftime(
        '%Y%m%d_%H%M', time.localtime())) + '.pkl')
    with open(fname, 'wb') as fw:
        pickle.dump(TRAIN_DATA, fw)

    fname = os.path.join(output_dir, 'train_data_{}.json'.format(time.strftime('%Y%m%d_%H%M', time.localtime())))
    with open(fname, 'w') as fw:
        json.dump(TRAIN_DATA, fw, ensure_ascii=False, indent=2)

    fname = os.path.join(output_dir, 'test_data_' + str(time.strftime(
        '%Y%m%d_%H%M', time.localtime())) + '.pkl')
    with open(fname, 'wb') as fw:
        test_ls = []
        con = 0
        q_ls = []
        for td in TRAIN_DATA:
            t_ls = []
            for t in td[1].get('entities'):
                t_ls.append(tuple((td[0][int(t[0]):int(t[1])], t[2])))
            test_ls.append(t_ls)
            q_ls.append(td[0])
            con += len(t_ls)
        test_tpl = tuple((con, q_ls, test_ls))
        pickle.dump(test_tpl, fw)

if __name__ == '__main__':
    st = time.time()
    plac.call(main)
    ed = time.time()
    print(' Cost: %.3f' % (ed - st), ' secs...')
