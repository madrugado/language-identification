import argparse

import numpy as np
from keras.models import load_model
import sentencepiece as spm
from keras_preprocessing.sequence import pad_sequences

from utils import lang_code_variants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--batch-size", type=int, default=256)
    parser.add_argument('-s', "--seq-len", type=int, default=256)
    args = parser.parse_args()

    model = load_model("task2.h5")

    sp = spm.SentencePieceProcessor()
    sp.Load("langid.brpt.model")

    test_data = []
    with open("langid-variants.test") as f:
        for line in f:
            test_data.append(sp.encode_as_ids(line))

    test_data = pad_sequences(test_data, args.seq_len)

    preds = model.predict(test_data, args.batch_size)
    preds = np.argmax(preds, axis=1)

    inv_lang_code = {v: k for k, v in lang_code_variants}
    for val in preds.tolist():
        print(inv_lang_code[val])
