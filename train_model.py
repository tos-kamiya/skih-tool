import sys
import os
import re
import random
import pickle
from collections import Counter

import docopt

import numpy as np


IDENTIFIER_PREFIX = '$I:'
OOV_TOKEN = '$OOV'
# SENTENCE_SEPS = ('$P:;', '$P:}', '$P:{')
SENTENCE_SEPS = ('$P:;', )


def eprint(s):
    print(s, file=sys.stderr, flush=True)


def identifier_split_java(token):
    token = token.rstrip()
    if token.startswith('@'):  # such as '@Override'  # java specific
        s2 = ['@'] + re.findall(r'[A-Z]+[a-z]*|[a-z]+|[:]', token[1:])
    else:
        s2 = re.findall(r'[A-Z]+[a-z]*|[a-z]+|[:]', token)  # 'default:' -> 'default', ':'
    if token.endswith('.*'):   # such as 'import java.util.*'  # java specific
        s2.append('*')
    return s2


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


def read_tseq(fp, identifier_split_func):
    with open(fp) as inp:
        L = inp.read()
    spec, text = L.split('\t')
    tseq = to_tseq(text.split(' '), identifier_split_func)
    return spec, tseq


def tseq_file_paths_from_dir_iter(train_dir, tseq_ext):
    for cur_dir, dirs, files in os.walk(train_dir):
        dirs.sort()  # for reproduceability
        for f in files:
            if f.endswith(tseq_ext):
                fp = os.path.join(cur_dir, f)
                yield fp


def read_tseq_from_dir_iter(train_dir, tseq_ext, identifier_split_func):
    for fp in tseq_file_paths_from_dir_iter(train_dir, tseq_ext):
        spec, tseq = read_tseq(fp, identifier_split_func)
        yield fp, spec, tseq


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


def gen_posi_nega_data_java(tseqwon, liseq, seq_length):
    # positive data -> sequence ends with ($P:; or $P:} or $P:{), followed by $C
    # negative data -> sequence ends with ($P:; or $P:} or $P:{), followed by other than $C
    min_distance = seq_length * 2

    len_tseqwon = len(tseqwon)
    posi_up_seqs = []
    posi_dn_seqs = []
    distance_from_last = min_distance
    for i, t in enumerate(tseqwon):
        distance_from_last += 1
        if t in SENTENCE_SEPS and distance_from_last >= min_distance and \
                i + 1 < len_tseqwon and tseqwon[i + 1] == '$C' and liseq[i] == liseq[i + 1]:
            se = i + 1
            si = max(0, se - seq_length)
            sp = se + seq_length
            posi_up_seqs.append(tseqwon[si:se])
            ds = tseqwon[se + 1:sp + 1]
            ds.reverse()
            posi_dn_seqs.append(ds)
            distance_from_last = 0

    nega_up_seqs = []
    nega_dn_seqs = []
    distance_from_last = min_distance
    for i, t in enumerate(tseqwon):
        distance_from_last += 1
        if t in SENTENCE_SEPS and distance_from_last >= min_distance and \
                not(i + 1 < len_tseqwon and tseqwon[i + 1] == '$C' and liseq[i] == liseq[i + 1]):
            se = i + 1
            si = max(0, se - seq_length)
            sp = se + seq_length
            nega_up_seqs.append(tseqwon[si:se])
            ds = tseqwon[se:sp]
            ds.reverse()
            nega_dn_seqs.append(ds)
            distance_from_last = 0

    assert len(posi_up_seqs) == len(posi_dn_seqs)
    assert len(nega_up_seqs) == len(nega_dn_seqs)

    return posi_up_seqs, posi_dn_seqs, nega_up_seqs, nega_dn_seqs


def gen_posi_nega_data_default(tseqwon, liseq, seq_length):
    min_distance = seq_length * 2

    len_tseqwon = len(tseqwon)
    posi_up_seqs = []
    posi_dn_seqs = []
    distance_from_last = min_distance
    for i, t in enumerate(tseqwon):
        distance_from_last += 1
        if distance_from_last >= min_distance and \
                i + 1 < len_tseqwon and tseqwon[i + 1] == '$C' and liseq[i] == liseq[i + 1]:
            se = i + 1
            si = max(0, se - seq_length)
            sp = se + seq_length
            posi_up_seqs.append(tseqwon[si:se])
            ds = tseqwon[se + 1:sp + 1]
            ds.reverse()
            posi_dn_seqs.append(ds)
            distance_from_last = 0

    nega_up_seqs = []
    nega_dn_seqs = []
    distance_from_last = min_distance
    for i, t in enumerate(tseqwon):
        distance_from_last += 1
        if distance_from_last >= min_distance and i + 1 < len_tseqwon and liseq[i] != liseq[i + 1]:
            se = i + 1
            si = max(0, se - seq_length)
            sp = se + seq_length
            nega_up_seqs.append(tseqwon[si:se])
            ds = tseqwon[se:sp]
            ds.reverse()
            nega_dn_seqs.append(ds)
            distance_from_last = 0

    assert len(posi_up_seqs) == len(posi_dn_seqs)
    assert len(nega_up_seqs) == len(nega_dn_seqs)

    return posi_up_seqs, posi_dn_seqs, nega_up_seqs, nega_dn_seqs


def get_posi_nega_data_from_dir(lang, d, tseq_ext, seq_length, remove_files_wo_comment=False):
    if lang == 'java':
        gen_posi_nega_data = gen_posi_nega_data_java
        identifier_split_func = identifier_split_java
    else:
        gen_posi_nega_data = gen_posi_nega_data_default
        identifier_split_func = identifier_split_default

    posi_up_tseqs = []
    posi_dn_tseqs = []
    nega_up_tseqs = []
    nega_dn_tseqs = []

    for fp, spec, tseq in read_tseq_from_dir_iter(d, tseq_ext, identifier_split_func):
        if remove_files_wo_comment and '$C' not in tseq:
            continue  # for fp, spec, tseq
        tseqwon, liseq = split_to_tseqwon_and_liseq(tseq)
        puts, pdts, nuts, ndts = gen_posi_nega_data(tseqwon, liseq, seq_length)
        if remove_files_wo_comment and not puts:
            continue  # for fp, spec, tseq
        posi_up_tseqs.extend(puts)
        posi_dn_tseqs.extend(pdts)
        nega_up_tseqs.extend(nuts)
        nega_dn_tseqs.extend(ndts)
    assert len(posi_up_tseqs) == len(posi_dn_tseqs)
    assert len(nega_up_tseqs) == len(nega_dn_tseqs)
    return posi_up_tseqs, posi_dn_tseqs, nega_up_tseqs, nega_dn_tseqs



def resample_shuffle_posi_nega(posi_us, posi_ds, nega_us, nega_ds):
    assert len(posi_us) == len(posi_ds)
    assert len(nega_us) == len(nega_ds)
    len_smaller = min(len(posi_us), len(nega_us))
    nud = random.sample([ud for ud in zip(nega_us, nega_ds)], len_smaller)
    nega_us = [ud[0] for ud in nud]
    nega_ds = [ud[1] for ud in nud]
    nud = None
    uud = random.sample([ud for ud in zip(posi_us, posi_ds)], len_smaller)
    posi_us = [ud[0] for ud in uud]
    posi_ds = [ud[1] for ud in uud]
    uud = None
    return posi_us, posi_ds, nega_us, nega_ds



def build_nn(params):
    seq_length, vocabulary_size, layers, embedding_dim, upside_dim, downside_dim, lr, dropout = \
        params['seq_length'], params['vocabulary_size'], params['layers'], params['embedding_dim'], params['upside_dim'], params['downside_dim'], params['lr'], params['dropout']

    from tensorflow.keras import Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GRU, Dense, Embedding, concatenate

    embedding = Embedding(input_dim=vocabulary_size, input_length=seq_length, output_dim=embedding_dim, mask_zero=True)
    upsideInput = Input(shape=(seq_length, ), name='upside_inp')
    upside_i = embedding(upsideInput)
    for i in range(layers):
        upside_i = GRU(upside_dim, return_sequences=i < layers - 1, name='upside_%d' % (i + 1), dropout=dropout)(upside_i)
    downsideInput = Input(shape=(seq_length, ), name='downside_inp')
    downside_i = embedding(downsideInput)
    for i in range(layers):
        downside_i = GRU(downside_dim, return_sequences=i < layers - 1, name='downside_%d' % (i + 1), dropout=dropout)(downside_i)
    output = Dense(1, activation='sigmoid')(concatenate([upside_i, downside_i]))

    model = Model(
        inputs=[upsideInput, downsideInput],
        outputs=[output]
    )
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model


def scan_dirs(lang, dirs, tseq_ext, seq_length, word_to_dir_count=None, resample_and_shuffle=True):
    posi_up_tseqs = []
    posi_dn_tseqs = []
    nega_up_tseqs = []
    nega_dn_tseqs = []

    for i, d in enumerate(dirs):
        word_set = set()

        pus, pds, nus, nds = get_posi_nega_data_from_dir(lang, d, tseq_ext, seq_length, 
                remove_files_wo_comment=True)
        posi_up_tseqs.extend(pus)
        posi_dn_tseqs.extend(pds)
        nega_up_tseqs.extend(nus)
        nega_dn_tseqs.extend(nds)

        if word_to_dir_count is not None:
            for s in pus:
                word_set.update(s)
            for s in pds:
                word_set.update(s)
            for s in nus:
                word_set.update(s)
            for s in nds:
                word_set.update(s)
            for w in word_set:
                word_to_dir_count[w] += 1

    if resample_and_shuffle:
        posi_up_tseqs, posi_dn_tseqs, nega_up_tseqs, nega_dn_tseqs = resample_shuffle_posi_nega(posi_up_tseqs, posi_dn_tseqs,nega_up_tseqs, nega_dn_tseqs)
    
    return posi_up_tseqs, posi_dn_tseqs, nega_up_tseqs, nega_dn_tseqs


def to_xy(posi_up_tseqs, posi_dn_tseqs, nega_up_tseqs, nega_dn_tseqs, seq_length, tokenizer, pad_sequences, remove_oov_tokens=False):
    if remove_oov_tokens:
        def remove_burst_oov_tokens_in_iseq(iseq):
            r = []
            for i in iseq:
                if i == 1 and r and r[-1] == 1:  # 1 is oov_token's index
                    continue
                r.append(i)
            return r

        posi_up_iseqs = [remove_burst_oov_tokens_in_iseq(iseq) for iseq in tokenizer.texts_to_sequences(posi_up_tseqs)]
        posi_dn_iseqs = [remove_burst_oov_tokens_in_iseq(iseq) for iseq in tokenizer.texts_to_sequences(posi_dn_tseqs)]
        nega_up_iseqs = [remove_burst_oov_tokens_in_iseq(iseq) for iseq in tokenizer.texts_to_sequences(nega_up_tseqs)]
        nega_dn_iseqs = [remove_burst_oov_tokens_in_iseq(iseq) for iseq in tokenizer.texts_to_sequences(nega_dn_tseqs)]
    else:
        posi_up_iseqs = tokenizer.texts_to_sequences(posi_up_tseqs)
        posi_dn_iseqs = tokenizer.texts_to_sequences(posi_dn_tseqs)
        nega_up_iseqs = tokenizer.texts_to_sequences(nega_up_tseqs)
        nega_dn_iseqs = tokenizer.texts_to_sequences(nega_dn_tseqs)

    pc, nc = len(posi_up_iseqs), len(nega_up_iseqs)
    xup = pad_sequences(posi_up_iseqs + nega_up_iseqs, seq_length)
    posi_up_iseqs = posi_up_iseqs = None
    xdn = pad_sequences(posi_dn_iseqs + nega_dn_iseqs, seq_length)
    posi_dn_iseqs = posi_dn_iseqs = None
    y = np.concatenate([np.ones(pc), np.zeros(nc)])

    return xup, xdn, y, pc, nc


__doc__ = """Make a model from code.

Usage:
  {argv0} [options] -l LANG -m MODEL -e EXT (<dir>...|-d DIRLIST)

Option:
  -l --language=<lang>  Programming language.
  -m MODEL      File name body of model file. e.g. `model` -> model.hdf5, model.pickle
  -e EXT        Extension of input token-sequence files. e.g. `.csin_tseq`.
  -d DIRLIST    List of directories (one dir per line).
  --seq-length=LEN      Length of token sequence to be learned [default: 100].
  --trials=NUM          Number of trials [default: 50].
  --verbose
"""


def main():
    args = docopt.docopt(__doc__)

    tseq_ext = args['-e']
    train_dirs = args['<dir>']
    if args['-d']:
        assert not train_dirs
        with open(args['-d']) as inp:
            train_dirs = [l.rstrip() for l in inp.readlines()]
    output_model = args['-m']
    seq_length = int(args['--seq-length'])
    # assert args['--remove-files-wo-comment'] in (None, 'False', 'True', 'false', 'true', '0', '1')
    # remove_files_wo_comment = args['--remove-files-wo-comment'] in ('True', 'true', '1')
    number_of_trials = int(args['--trials'])
    verbose = args['--verbose']
    language = args['--language']

    for i in range(len(train_dirs)):
        if train_dirs[i] == '/':
            valid_dirs = train_dirs[i+1:]
            train_dirs = train_dirs[:i]
            assert valid_dirs
            break  # for i
    else:
        valid_dirs = []

    if not valid_dirs:
        sys.exit("Error: no validation data dir")

    assert train_dirs  # should not be empty

    import tensorflow.keras.backend as K
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.models import Model, load_model

    import optuna

    if verbose:
        eprint("> Read tokens of training data...")

    def fit_on_text(posi_up_tseqs, posi_dn_tseqs, nega_up_tseqs, nega_dn_tseqs, tokenizer, oov_token_set=None):
        if oov_token_set:
            def snap_oov_token(tseq):
                return [t for t in tseq if t not in oov_token_set]
            
            tokenizer.fit_on_texts([snap_oov_token(tseq) for tseq in posi_up_tseqs])
            tokenizer.fit_on_texts([snap_oov_token(tseq) for tseq in posi_dn_tseqs])
            tokenizer.fit_on_texts([snap_oov_token(tseq) for tseq in nega_up_tseqs])
            tokenizer.fit_on_texts([snap_oov_token(tseq) for tseq in nega_dn_tseqs])
        else:
            tokenizer.fit_on_texts(posi_up_tseqs)
            tokenizer.fit_on_texts(posi_dn_tseqs)
            tokenizer.fit_on_texts(nega_up_tseqs)
            tokenizer.fit_on_texts(nega_dn_tseqs)

    word_to_dir_count = Counter()
    train_tseqs = scan_dirs(language, train_dirs, tseq_ext, seq_length, word_to_dir_count=word_to_dir_count, resample_and_shuffle=True)
    # for w, dc in sorted(word_to_dir_count.items(), key=lambda wdc: wdc[1], reverse=True):
    #     print("w, dc = %s, %d" % (w, dc))
    words_appears_only_one_dir = [w for w, dc in word_to_dir_count.items() if dc <= 1]
    oov_token_set = frozenset(words_appears_only_one_dir)
    words_appears_only_one_dir = None

    tokenizer = Tokenizer(oov_token=OOV_TOKEN, num_words=10000)
    fit_on_text(*train_tseqs, tokenizer, oov_token_set=oov_token_set)
    xup, xdn, y, pc, nc = to_xy(*train_tseqs, seq_length, tokenizer, pad_sequences)
    train_tseqs = None

    vocabulary_size = len(tokenizer.word_index) + 1

    if verbose:
        eprint("Info: num train dirs\t%d" % len(train_dirs))
        eprint("Info: num positive fragments\t%d" % pc)
        eprint("Info: num negative fragments\t%d" % nc)

    if valid_dirs:
        if verbose:
            eprint("> Read tokens of validation data")

        val_tseqs = scan_dirs(language, valid_dirs, tseq_ext, seq_length, resample_and_shuffle=True)
        val_xup, val_xdn, val_y, pc, nc = to_xy(*val_tseqs, seq_length, tokenizer, pad_sequences)
        val_tseqs = None

        if pc == 0 or nc == 0:
            exit("Error: not enough fragments found as validation data")
    else:
        val_xup = val_xdn = val_y = None
    
    if verbose:
        eprint("> Study...")

    def build_model(trial):
        layers = 2
        embedding_dim = 24
        upside_dim = 160
        downside_dim = 16
        lr = trial.suggest_uniform('lr', 0.0001, 0.001)
        dropout = trial.suggest_uniform('dropout', 0.01, 0.1)
        batch_size = trial.suggest_int('batch_size', 1024, 4096, 256)

        params = {
            "seq_length": seq_length, "vocabulary_size": vocabulary_size, "layers": layers, "embedding_dim": embedding_dim, 
            "upside_dim": upside_dim, "downside_dim": downside_dim, "lr": lr, "dropout": dropout,
            "batch_size": batch_size,
        }

        return build_nn(params), params

    best_model_in_trial = [None]
    best_model_params_in_trial = [None]

    def objective(trial):
        K.clear_session()
        model, params = build_model(trial)

        model_temp_filename = 'tmp-%08d.h5' % random.randrange(1, 100000000)
        batch_size = params["batch_size"]
        epochs = 40
        
        model.fit(
            [xup, xdn], y,
            batch_size=batch_size, epochs=epochs,
            validation_data=([val_xup, val_xdn], val_y),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='min'),
                ModelCheckpoint(model_temp_filename, monitor='val_loss', save_best_only=True, 
                        save_wait_only=False, mode='min', peroid=1)
            ],
            verbose=2
        )

        best_model_in_trial[0] = model = load_model(model_temp_filename)
        best_model_params_in_trial[0] = params
        os.remove(model_temp_filename)
        score = model.evaluate([val_xup, val_xdn], val_y, verbose=0)
        eprint("val_loss: %g val_accuracy: %g" % (score[0], score[1]))
        return score[0]  # val_loss

    def save_model_callback(study, trial):
        if study.best_trial != trial:
            return
        if verbose:
            eprint("> Save model.")
        model = best_model_in_trial[0]
        model.save(output_model + '.hdf5')
        params = best_model_params_in_trial[0]
        params['tokenizer'] = tokenizer
        with open(output_model + '.pickle', 'wb') as outp:
            pickle.dump(params, outp, protocol=pickle.HIGHEST_PROTOCOL)
    
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=number_of_trials, callbacks=[save_model_callback])

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # model.save(output_model + '.hdf5')

    # with open(output_model + '.pickle', 'wb') as outp:
    #     pickle.dump(tokenizer, outp, protocol=pickle.HIGHEST_PROTOCOL)

    # if verbose:
    #     from sklearn.metrics import confusion_matrix
    #     pred = (model.predict([val_xup, val_xdn]) >= 0.5).astype("int32")
    #     print(confusion_matrix(val_y, pred))


if __name__ == '__main__':
    main()

