import os
import sys
import pickle
from itertools import zip_longest

import docopt
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
import numpy as np

from gen_tokenseqs import read_lines_safe, scan_tokens, normalize_token_seq
from gen_fragment import IDENTIFIER_PREFIX, OOV_TOKEN, SENTENCE_SEPS
from gen_fragment import identifier_split_java, identifier_split_python, identifier_split_default
from gen_fragment import to_tseq, split_to_tseqwon_and_liseq


script_dir = os.path.dirname(os.path.realpath(__file__))


def code_wo_comments(code, lexer):
    tokenstrs_wo_comments = []
    token_it = lexer.get_tokens_unprocessed(code)
    for idx, tt, ts in token_it:
        if tt in Token.Comment:
            tokenstrs_wo_comments.append('\n' * ts.count('\n'))
        else:
            tokenstrs_wo_comments.append(ts)
    return ''.join(tokenstrs_wo_comments)


def enumerate_lines(lines):
    for i, l in enumerate(lines):
        yield [i], l


def enumerate_lines_w_merging_empty_lines(lines):
    lines = list(lines)
    line_indices = []
    for i, (lcur, lnext) in enumerate(zip_longest(lines, lines[1:])):
        if lcur == lnext == '':
            line_indices.append(i)
        else:
            line_indices.append(i)
            yield line_indices, lcur
            line_indices = []
    assert line_indices == []


_pad_sequences_func = [None]


def calc_line_comment_probability(iseq, idx, model, seq_length, no_count_token_set=None):
    pad_sequences = _pad_sequences_func[0]
    no_count_token_set = frozenset([]) if no_count_token_set is None else no_count_token_set

    up_iseq = []
    for i in range(idx, 0, -1):
        t = iseq[i]
        if t not in no_count_token_set:
            up_iseq.append(t)
        if len(up_iseq) >= seq_length:
            break  # for i
    up_iseq.reverse()

    dn_iseq = []
    for i in range(idx + 1, len(iseq)):
        t = iseq[i]
        if t not in no_count_token_set:
            dn_iseq.append(t)
        if len(dn_iseq) >= seq_length:
            break  # for i
    dn_iseq.reverse()

    iseqs = pad_sequences([up_iseq, dn_iseq], seq_length)
    xup = np.reshape(iseqs[0], (1, seq_length))
    xdn = np.reshape(iseqs[1], (1, seq_length))
    predict = model([xup, xdn]).numpy()[0]
    return predict


def calc_probability(tokens, tokenizer, model, seq_length, predict_wo_comment_tokens=False):
    tseq = to_tseq(normalize_token_seq(tokens), identifier_split_java)

    tseqwon, liseq = split_to_tseqwon_and_liseq(tseq)
    sentence_sep_poss = [i for i, t in enumerate(tseqwon) if t in SENTENCE_SEPS]
    iseq = tokenizer.texts_to_sequences([tseqwon])[0]
    assert len(iseq) == len(tseqwon)
    if predict_wo_comment_tokens:
        comment_token_i = tokenizer.texts_to_sequences([['$C']])[0][0]
        no_count_token_set = frozenset([comment_token_i])
    else:
        no_count_token_set = frozenset([])

    li2p = dict()
    for i in sentence_sep_poss:
        assert tseqwon[i] in SENTENCE_SEPS
        li2p[liseq[i]] = calc_line_comment_probability(iseq, i, model, seq_length,
                no_count_token_set=no_count_token_set)

    return li2p


__doc__ = """Apply a model to code.

Usage:
  {argv0} [options] -l LANG [-m MODEL] (-n|-p THRES|-t NUM) <sourcefile>...

Option:
  -l --language=<lang>          Programming language.
  -m MODEL                      File name body of model file. e.g. `model` -> model.hdf5, model.pickle
  -n --show-probability         Show number of probability.
  -p THRES                      Threshold of probability of comment.
  -t NUM                        Top-N probability of comment.
  -C --remove-original-comments     Remove original comments from source.
  -w --predict-wo-comment-tokens    Remove comment tokens for prediction.
"""


def main():
    args = docopt.docopt(__doc__)
    language = args['--language']
    input_model = args['-m']
    source_files = args['<sourcefile>']
    threshold = float(args['-p']) if args['-p'] else None
    top_n = int(args['-t']) if args['-t'] else None
    if top_n is not None and top_n < 1:
        sys.exit('Error: -t NUM should be >= 1.')
    show_probability = args['--show-probability']
    remove_original_comments = args['--remove-original-comments']
    predict_wo_comment_tokens = args['--predict-wo-comment-tokens']
    
    if not input_model:
        input_model = os.path.join(script_dir, language)

    lexer = get_lexer_by_name(language)
    if lexer is None:
        sys.exit("Error: no language specific settings found for language: %s" % language)

    from tensorflow.python.keras.preprocessing.sequence import pad_sequences
    _pad_sequences_func[0] = pad_sequences
    from build_nn import build_nn

    with open(input_model + ".pickle", 'rb') as inp:
        model_params = pickle.load(inp)
    model = build_nn(model_params)
    model.load_weights(input_model + '.hdf5')

    upside_seq_length = downside_seq_length = seq_length = model_params['seq_length']
    tokenizer = model_params['tokenizer']

    outp = sys.stdout
    for fname in source_files:
        lines = read_lines_safe(fname)
        code = '\n'.join(lines)
        tokens = scan_tokens(fname, code, lexer)

        li2p = calc_probability(tokens, tokenizer, model, seq_length, 
                predict_wo_comment_tokens=predict_wo_comment_tokens)

        if top_n is not None:
            if len(li2p) > top_n:
                ps = list(li2p.values())
                ps.sort(reverse=True)
                threshold = ps[top_n - 1]
                ps = None
            else:
                threshold = 0.0

        if remove_original_comments:
            lines = code_wo_comments(code, lexer).split('\n')
            el_func = enumerate_lines_w_merging_empty_lines
        else:
            el_func = enumerate_lines

        if show_probability:
            def line_format_func(line_indices, l):
                max_p = max(li2p.get(i, 0.0) for i in line_indices)
                if max_p > 0.0:
                    return "%.3f %s" % (max_p, l)
                else:
                    return "%5s %s" % ("", l)
        else:
            def line_format_func(line_indices, l):
                m = '*' if any(li2p.get(i, 0.0) >= threshold for i in line_indices) else ' '
                return "%s %s" % (m, l)

        print("- %s" % fname, file=outp)
        for line_indices, l in el_func(lines):
            print(line_format_func(line_indices, l), file=outp)


if __name__ == '__main__':
    main()
