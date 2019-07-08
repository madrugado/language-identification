import argparse

import numpy as np
from keras.models import load_model
import sentencepiece as spm
from keras_preprocessing.sequence import pad_sequences

from task3_utils import unbpe, lang_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", type=str, default='enes')
    parser.add_argument("-f", "--filename", type=str, default="data/dev_tweets.tsv")
    parser.add_argument('-s', "--seq-len", type=int, default=256)
    parser.add_argument('-b', "--batch-size", type=int, default=256)

    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.Load("langid.{}.model".format(args.langs))

    tweets = []
    tweet_texts = []
    with open(args.filename) as f:
        for line in f:
            tweets.append(sp.encode_as_ids(line))
            tweet_texts.append(sp.encode_as_pieces(line))

    model = load_model("task3.h5")
    tweets = pad_sequences(tweets, args.seq_len)
    preds = model.predict(tweets, batch_size=args.batch_size)
    preds = np.argmax(preds, axis=-1)

    inv_lang_code = dict([(p[1], p[0]) for p in lang_code.items()])
    for i in range(len(tweet_texts)):
        raw_output = preds[i, :len(tweet_texts[i])].tolist()
        raw_labels = [inv_lang_code[p] for p in raw_output]
        print(unbpe(tweet_texts[i], raw_labels))
