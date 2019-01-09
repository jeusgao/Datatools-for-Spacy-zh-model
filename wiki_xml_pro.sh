#!/bin/bash
# preprocess data

# Traditional Chinese to Simplified Chinese
echo "opencc: Traditional Chinese to Simplified Chinese..."
#time opencc -i wiki.zh.txt -o wiki.zh.chs.txt -c zht2zhs.ini
time opencc -i zhwiki_latest.txt -o zhwiki_latest_chs.txt -c t2s.json

# Cut words
echo "jieba: Cut words..."
time python -m jieba -d ' ' zhwiki_latest_chs.txt > zhwiki_seg.txt

# Change encode
echo "iconv: ascii to utf-8..."
time iconv -c -t UTF-8 < zhwiki_seg.txt > zhwiki_seg_utf.txt
