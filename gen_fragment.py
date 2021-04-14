import re


IDENTIFIER_PREFIX = '$I:'
OOV_TOKEN = '$OOV'
# SENTENCE_SEPS = ('$P:;', '$P:}', '$P:{')
SENTENCE_SEPS = ('$P:;', )


def identifier_split_java(token):
    token = token.rstrip()
    if token.startswith('@'):  # such as '@Override'  # java specific
        s2 = ['@'] + re.findall(r'[A-Z]+[a-z]*|[a-z]+|[:]', token[1:])
    else:
        s2 = re.findall(r'[A-Z]+[a-z]*|[a-z]+|[:]', token)  # 'default:' -> 'default', ':'
    if token.endswith('.*'):   # such as 'import java.util.*'  # java specific
        s2.append('*')
    return s2


def identifier_split_python(token):
    token = token.rstrip()
    if token.startswith('__'):  # special method name (should be treated as a reserved word?)
        return [token]
    s2 = re.findall(r'[A-Z]+[a-z]*|[a-z]+', token)
    return [w.lower() for w in s2]


def identifier_split_default(token):
    token = token.rstrip()
    s2 = re.findall(r'[A-Z]+[a-z]*|[a-z]+', token)
    return [w.lower() for w in s2]


def to_tseq(tokens, identifier_split_func):
    len_IDENTIFIER_PREFIX = len(IDENTIFIER_PREFIX)
    r = []
    for t in tokens:
        if t.startswith(IDENTIFIER_PREFIX):
            r.append(IDENTIFIER_PREFIX)
            r.extend(identifier_split_func(t[len_IDENTIFIER_PREFIX:]))
        else:
            r.append(t)
    return r


def split_to_tseqwon_and_liseq(tseq):
    tseqwon = []
    liseq = []

    li = 0
    for t in tseq:
        if t == '$$':
            li += 1
        else:
            tseqwon.append(t)
            liseq.append(li)
    
    return tseqwon, liseq
