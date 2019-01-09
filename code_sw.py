# -*- coding: utf-8 -*-
with open('SogouR.txt', 'rb') as fr:
    ol = fr.read().decode('UTF-8', 'strict')


with open('wrds_grp.txt', 'w') as fw:
    fw.write(ol)
    fw.flush()
