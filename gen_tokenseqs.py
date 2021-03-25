import os
import sys
import re
from fnmatch import fnmatch

from pygments.lexers import get_lexer_by_name
from pygments.token import Token
import docopt

from gen_tokenseqs_langspecific import LANG_SPEC_TABLE


def read_lines_safe(filename):
    with open(filename, encoding='utf-8', errors='replace') as inp:
        text = inp.read()
        lines = text.split('\n')
        if lines[-1] == '':
            lines = lines[:-1]
        return lines


def scan_tokens(filename, code, lexer):
    _ = filename
    prev_idx = 0
    tokens = []
    token_it = lexer.get_tokens_unprocessed(code)
    for idx, tt, ts in token_it:
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
            ts = re.sub(r'[\r\n\t]', '', ts).strip()  # remove space chars (if they were) in token (a buggy behavior of lexer)
            if tt in Token.Keyword:
                tokens.append("$K:" + ts)
            elif tt in Token.Name:
                tokens.append("$I:" + ts)  # identifier
            elif tt in Token.Operator:
                tokens.append("$O:" + ts)
            elif tt in Token.Punctuation:
                tokens.append("$P:" + ts)
        prev_idx = idx
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

    langspec = LANG_SPEC_TABLE.get(language, None)
    if langspec is None:
        sys.exit("Error: no language specific settings found for language: %s" % language)
    lexer = get_lexer_by_name(langspec.lexer_name)

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
