import os
import sys
import re
from fnmatch import fnmatch

from pygments.lexers import get_lexer_by_name
from pygments.token import Token
import docopt


def read_lines_safe(filename):
    with open(filename, encoding='utf-8', errors='replace') as inp:
        text = inp.read()
        lines = text.split('\n')
        if lines[-1] == '':
            lines = lines[:-1]
        return lines


PSEUDO_SENTENCE_SEP = '$P:;'
PSEUDO_DEDENT = '$}'


def scan_tokens(filename, code, lexer):
    _ = filename
    prev_idx = 0
    prev_tt_ts = None, None
    tokens = []
    if lexer.name == 'Python':
        code = re.sub('\n[ \t]+\n', '\n\n', code)
    token_it = lexer.get_tokens_unprocessed(code)
    cur_indent = ''
    for idx, tt, ts in token_it:
        if lexer.name == 'Python':
            is_empty_line = False
            if prev_tt_ts[0] in Token.Text and prev_tt_ts[1] == '\n':
                if tt in Token.Text and ts == '\n':
                    pass  # empty lines can not change depth of indentation
                else:
                    if tt in Token.Text and re.match('[ \t]+', ts):
                        indent = ts
                    else:
                        indent = ''
                    if len(indent) < len(cur_indent):
                        tokens.append(PSEUDO_DEDENT)
                    cur_indent = indent
            if tt in Token.Text and ts == '\n':
                line_has_statement = False
                for t in tokens[::-1]:
                    if t == '\n':
                        break  # for t
                    if t not in ['$C', '$S', PSEUDO_SENTENCE_SEP]:
                        line_has_statement = True
                if line_has_statement:
                    if tokens[-1] == '$C':
                        tokens.insert(len(tokens) - 1, PSEUDO_SENTENCE_SEP)
                    else:
                        tokens.append(PSEUDO_SENTENCE_SEP)
        tokens.extend(['\n'] * code[prev_idx:idx].count('\n'))
        if tt in Token.Literal.String:
            if tokens and tokens[-1] != "$S":
                tokens.append("$S")  # it seems single string literal is split into multiple tokens.
        elif tt in Token.Literal.Number:
            tokens.append("$N:" + ts)
        elif tt in Token.Literal:
            tokens.append("$L")
        elif tt in Token.Comment:
            tokens.append("$C")
        else:
            tts = re.sub(r'[\r\n\t]', '', ts).strip()  # remove space chars (if they were) in token (a buggy behavior of lexer)
            if tt in Token.Keyword:
                tokens.append("$K:" + tts)
            elif tt in Token.Name:
                tokens.append("$I:" + tts)  # identifier
            elif tt in Token.Operator:
                tokens.append("$O:" + tts)
            elif tt in Token.Punctuation:
                tokens.append("$P:" + tts)
        prev_idx = idx
        prev_tt_ts = tt, ts
    else:
        tokens.extend(['\n'] * code[prev_idx:].count('\n'))
    return tokens


def to_fragment(tokenid_seq):
    tokenid_itemset = list(set(tokenid_seq))
    tokenid_itemset.sort()
    return tokenid_itemset


def normalize_token_seq(tokens):
    assert ' ' not in tokens
    tokens = [('$$' if t == '\n' else t) for t in tokens]
    return tokens


def do_lexical_analysis(outp, f, lexer):
    lines = read_lines_safe(f)
    tokens = scan_tokens(f, '\n'.join(lines), lexer)
    tokens = normalize_token_seq(tokens)
    print("%s\t%s" % (f, ' '.join(tokens)), file=outp)


def traverse_dir_iter(root_dir, file_name_specs):
    for cur_dir, dirs, files in os.walk(root_dir):
        for f in files:
            if any(fnmatch(f, s) for s in file_name_specs):
                yield os.path.join(cur_dir, f)


def select_file_iter(files, file_name_specs):
    for fp in files:
        f = os.path.basename(fp)
        if any(fnmatch(f, s) for s in file_name_specs):
            yield fp
        else:
            print("warning: skip file: %s" % repr(fp), file=sys.stderr)


__doc__ = """Generate token sequence.

Usage:
  {argv0} -l LANG [-o OUTPUT|-p PAT] <file>...
  {argv0} -l LANG [-o OUTPUT|-p PAT] -d DIR
  {argv0} --help

Options:
  -l --language=<lang>          Programming language.
  -d --source-dir=<dir>         Root directory of target source files.
  -o <output>                   Output file.
  -p <pat>                      Pattern of output file. e.g. '%b.csin_tseq'
""".format(argv0=os.path.basename(sys.argv[0]))


def main():
    args = docopt.docopt(__doc__)

    language = args['--language']
    target_files = args['<file>']
    target_dir = args['--source-dir']
    if target_dir and target_dir.endswith(os.path.sep):
        target_dir = target_dir[:-len(os.path.sep)]  # strip trailing '/'
    output_file = args['-o']
    output_pattern = args['-p']

    if output_pattern:
        if output_pattern == '%b':
            sys.exit("error: pattern of output file not renaming (not including other than '%b'")
        if output_pattern.find('%b') < 0:
            sys.exit("error: pattern of output file not including '%b': %s" % repr(output_pattern))

    lexer = get_lexer_by_name(language)
    if lexer is None:
        sys.exit("Error: no language specific settings found for language: %s" % language)

    if target_dir:
        target_file_it = traverse_dir_iter(target_dir, lexer.filenames)
    else:
        target_file_it = select_file_iter(target_files, lexer.filenames)

    if output_pattern:
        for fp in target_file_it:
            fd, fb = os.path.split(fp)
            out_fb = output_pattern.replace('%b', fb)
            out_fp = os.path.join(fd, out_fb)
            try:
                with open(out_fp, 'w') as outp:
                    do_lexical_analysis(outp, fp, lexer)
            except FileNotFoundError as e:
                print("Warning: fail to open file: %s" % repr(fp), file=sys.stderr)
                try:
                    os.remove(out_fp)
                except:
                    pass
            except UnicodeEncodeError as e:
                print("Warning: invalid character in file: %s" % repr(fp), file=sys.stderr)
                try:
                    os.remove(out_fp)
                except:
                    pass
    elif output_file:
        with open(output_file, 'w') as outp:
            for fp in target_file_it:
                try:
                    do_lexical_analysis(outp, fp, lexer)
                except FileNotFoundError as e:
                    print("Warning: fail to open file: %s" % repr(fp), file=sys.stderr)
    else:
        for fp in target_file_it:
            do_lexical_analysis(sys.stdout, fp, lexer)


if __name__ == '__main__':
    main()
