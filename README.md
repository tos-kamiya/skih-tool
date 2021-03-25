# Skih-tool

A tool for predicting lines where end-of-sentence comments appear using DNN.

## Install dependencies

1. Install tensorflow for CPU or GPU, depending on your environment.
2. Install other dependencies listed in `requirements.txt`.

E.g., 

```sh
python3 -m pip install tensorflow
python3 -m pip install -r requirements.txt
```

## Run

Run the script `apply_model.py` with a programming language, a threshold (the larger the threshold, the less end-of-sentence comments are output), and a source file.

```
python3 apply_model.py -l <language> -p <threshold> <sourcefile>
```

The tensorflow log will be output to the standard error output, so redirect if you do not want to see the log.

E.g.,

```sh
python3 apply_model.py -l python -p 0.7 main.py 2> /dev/null
```

```sh
python3 apply_model.py -l java -p 0.7 Main.java 2> /dev/null
```
