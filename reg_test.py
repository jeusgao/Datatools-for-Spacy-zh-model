import re
a = ['知道', '知不知道', '了解']
b = ['有个', '一个', '那个']
c = ['什么是', '谁是', '啥是', '谁叫', '叫']
d = ['的', '的人', '的东西', '的地方']
e = ['是', '在', '去', '到', '的']
f = ['是谁', '谁', '说', '什么', '哪', '哪儿', '哪里', '哪个', '啥']
g = ['去', '东西', '国家', '地方']
h = ['的', '地', '得']
i = ['人', '城市']
j = ['呢', '啊', '哪', '吗', '呀', '吧', '那', '了']
cn_phrases_lemma = r'^(' + '|'.join(a) + ')?(' + '|'.join(b) + ')?(' + \
    '|'.join(c) + ')?(?P<subject>.+?)(' + \
    '|'.join(d) + ')?(' + '|'.join(e) + ')?(' + '|'.join(f) + ')?(' + \
    '|'.join(g) + ')?(' + '|'.join(h) + ')?(' + '|'.join(i) + ')?(' + \
    '|'.join(j) + ')?$'
# cn_phrases_lemma = r'^(知道|知不知道|了解)?(有个|一个|那个)?(什么是|谁是|啥是|谁叫|叫)?(?P<subject>.+?)(的|的人|的东西|的地方)?(是|在|去|到|的)?(是谁|谁|说|什么|哪|哪儿|哪里|哪个|啥)?(去|东西|国家|地方)?(的|地|得)?(人|城市)?(呢|啊|哪|吗|呀|吧|那|了)?$'
cn_regex_lemma = re.compile(cn_phrases_lemma, flags=re.IGNORECASE)

while True:
    qry = input("> ")
    print(re.sub(cn_regex_lemma, r'\g<subject>', qry))