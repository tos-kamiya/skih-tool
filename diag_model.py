import sys
import pickle

import docopt

from train_model import scan_dirs, to_xy


def eprint(s):
    print(s, file=sys.stderr, flush=True)


def calc_weighted_pre_rec_f1(cm, weight=None):
    assert weight in ('true', 'pred')

    n_p0 = cm[0, 0] + cm[1, 0]
    n_p1 = cm[0, 1] + cm[1, 1]
    n_t0 = cm[0, 0] + cm[0, 1]
    n_t1 = cm[1, 0] + cm[1, 1]

    pre_0 = cm[0, 0] / n_p0
    pre_1 = cm[1, 1] / n_p1

    pre_wp = (pre_0 * n_p0 + pre_1 * n_p1) / (n_p0 + n_p1)
    pre_wt = (pre_0 * n_t0 + pre_1 * n_t1) / (n_t0 + n_t1)

    rec_0 = cm[0, 0] / n_t0
    rec_1 = cm[1, 1] / n_t1

    rec_wp = (rec_0 * n_p0 + rec_1 * n_p1) / (n_p0 + n_p1)
    rec_wt = (rec_0 * n_t0 + rec_1 * n_t1) / (n_t0 + n_t1)

    f1_wp = 2 * pre_wp * rec_wp / (pre_wp + rec_wp)
    f1_wt = 2 * pre_wt * rec_wt / (pre_wt + rec_wt)

    if weight == 'pred':
        return pre_wp, rec_wp, f1_wp
    elif weight == 'true':
        return pre_wt, rec_wt, f1_wt
    else:
        assert False


__doc__ = """Diagnose/evaluate model.

Usage:
  {argv0} [options] summary -l LANG -m MODEL
  {argv0} [options] plot -l LANG -m MODEL -o OUTPUTPNGFILE
  {argv0} [options] eval -l LANG -m MODEL -e EXT <dir>...
  {argv0} [options] eval -l LANG -m MODEL -e EXT -d DIRLIST

Option:
  -l --language=<lang>  Programming language.
  -m MODEL      File name body of model file. e.g. `model` -> model.h5, model.pickle
  -e EXT        Extension of input token-sequence files. e.g. `.csin_tseq`
  -d DIRLIST    List of directories (one dir per line)
  -o FILE       Output.
"""


def main():
    args = docopt.docopt(__doc__)
    input_model = args['-m']
    tseq_ext = args['-e']
    test_dirs = args['<dir>']
    if args['-d']:
        assert not test_dirs
        with open(args['-d']) as inp:
            test_dirs = [l.rstrip() for l in inp.readlines()]
    language = args['--language']

    from train_model import build_nn

    with open(input_model + ".pickle", 'rb') as inp:
        model_params = pickle.load(inp)
    model = build_nn(model_params)
    model.load_weights(input_model + '.hdf5')

    upside_seq_length = downside_seq_length = seq_length = model_params['seq_length']

    if args['summary']:
        print("upside seq length\t%d" % upside_seq_length)
        print("downside seq length\t%d" % downside_seq_length)
        print(model.summary())
        return
    elif args['plot']:
        from keras.utils import plot_model
        output_png_file = args['-o']
        plot_model(model, show_shapes=True, to_file=output_png_file)
        return

    assert args['eval']

    if not test_dirs:
        sys.exit("Error: no test data dir")

    tokenizer = model_params['tokenizer']

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    test_tseqs = scan_dirs(language, test_dirs, tseq_ext, seq_length,  resample_and_shuffle=False)
    test_xup, test_xdn, test_y, pc, nc = to_xy(*test_tseqs, seq_length, tokenizer, pad_sequences)
    test_tseqs = None

    if pc == 0 or nc == 0:
        exit("Error: not enough fragments found as test data")

    score = model.evaluate([test_xup, test_xdn], test_y, verbose=0)
    print("test loss: %g" % score[0])
    print("test accuracy: %g" % score[1])

    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    pred = (model([test_xup, test_xdn]).numpy() >= 0.5).astype("int32")
    cm = confusion_matrix(test_y, pred)
    print("confusion matrix:")
    print(cm)

    print("precision score: %g" % precision_score(test_y, pred))
    print("recall score: %g" % recall_score(test_y, pred))
    print("f1 score: %g" % f1_score(test_y, pred))

    pre_w, rec_w, f1_w = calc_weighted_pre_rec_f1(cm, weight='pred')
    print("weighted precision score: %g" % pre_w)
    print("weighted recall score: %g" % rec_w)
    print("weighted f1 score: %g" % f1_w)

    # !! sklern's weighted scores are by 'true' class, not 'precition' class !!
    # print("weighted precision score: %g" % precision_score(test_y, pred, average='weighted'))
    # print("weighted recall score: %g" % recall_score(test_y, pred, average='weighted'))
    # print("weighted f1 score: %g" % f1_score(test_y, pred, average='weighted'))


if __name__ == '__main__':
    main()

