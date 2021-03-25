class LangSpec:
    def __init__(self, language, lexer_name, kinds, token_seq_preprocess):
        self.language = language
        self.lexer_name = lexer_name
        self.kinds = kinds
        self.token_seq_preprocess = token_seq_preprocess


def remove_definition_first_line_and_last_closing_brace_and_doubtful_comments(ts):
    open_brace_appear_first_line = False
    while ts and ts[0] != '':
        if ts[0].endswith(':{'):
            open_brace_appear_first_line = True
        del ts[0]

    if ts:
        last_token = ts[-1]
        if last_token.endswith(':}') and open_brace_appear_first_line:
            del ts[-1]
    
    # prevent from generating tokens from one-line comment "//..."
    for i in range(len(ts) - 1):
        if ts[i] == "$O:/" and ts[i + 1] == "$O:/":
            ts = ts[:i]
            break

    return ts


def remove_definition_first_line_and_last_closing_brace(ts):
    open_brace_appear_first_line = False
    while ts and ts[0] != '':
        if ts[0].endswith(':{'):
            open_brace_appear_first_line = True
        del ts[0]

    if ts:
        last_token = ts[-1]
        if last_token.endswith(':}') and open_brace_appear_first_line:
            del ts[-1]
    
    return ts


def remove_definition_first_line(ts):
    open_brace_appear_first_line = False
    while ts and ts[0] != '':
        if ts[0].endswith(':{'):
            open_brace_appear_first_line = True
        del ts[0]

    if ts:
        last_token = ts[-1]
        if last_token.endswith(':}') and open_brace_appear_first_line:
            del ts[-1]
    
    return ts


LANG_SPECS = [
    LangSpec('java', 'java', ['m'], remove_definition_first_line_and_last_closing_brace),
    LangSpec('c', 'c', ['f'], remove_definition_first_line_and_last_closing_brace_and_doubtful_comments),
    LangSpec('python', 'python', ['f'], remove_definition_first_line),
]

# LANG_SPEC_FALLBACK = LangSpec(None, ['f'], remove_definition_first_line)

LANG_SPEC_TABLE = dict((s.language, s) for s in LANG_SPECS)
