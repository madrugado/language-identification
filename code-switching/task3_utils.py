from collections import defaultdict

import numpy as np
from keras.utils import Sequence
import sentencepiece as spm
from keras.preprocessing.sequence import pad_sequences

from keras import Model, Sequential
from keras.constraints import MaxNorm
from keras.layers import Dense, Activation, Embedding, Input, concatenate, LSTM, GlobalMaxPooling1D, Conv1D, \
    Bidirectional, Masking, TimeDistributed
from tqdm import tqdm

lang_code = {'en': 0, 'es': 1, 'other': 2}


class CodeSwitchingGenerator(Sequence):
    def __init__(self, batch_size, seq_len, mode='train'):
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("langid.enes.model")

        with open("data/{}_data.tsv".format(mode)) as f:
            data = f.readlines()

        self.data = defaultdict(list)
        self.labels = defaultdict(list)
        for line in tqdm(data):
            twit_id, user_id, _, _, rest = line.split('\t', 4)
            token, lang = rest.rsplit('\t', 1)
            tokens = self.sp.encode_as_ids(token)
            self.data[int(twit_id)] += (tokens)
            self.labels[int(twit_id)] += ([lang_code[lang.strip()]] * len(tokens))

        self.twits = list(self.data.keys())
        self.clean_data()

    def clean_data(self):
        for key in self.twits:
            if not self.data[key]:
                del self.data[key]
                del self.labels[key]
        self.twits = list(self.data.keys())

    def __getitem__(self, index):
        temp_keys = self.twits[self.batch_size * index: (self.batch_size + 1) * index]
        batch = [self.data[key] for key in temp_keys]
        target = [self.labels[key] for key in temp_keys]

        return pad_sequences(batch, self.seq_len, padding='post', truncating='post'), \
            to_categorical(target, self.seq_len)

    def __len__(self):
        return len(self.twits) // self.batch_size


class Seq2SeqModel:
    def __init__(self, args):
        embs = np.load("embeddings.npy")

        self.model = Sequential()
        self.model.add(Embedding(args.vocab_size, args.emb_dim,
                                 mask_zero=True, name='word_emb',
                                 weights=embs, trainable=False))
        self.model.add(Bidirectional(LSTM(args.hidden_size, return_sequences=True),
                                     input_shape=(args.seq_len, 1)))
        self.model.add(TimeDistributed(Dense(args.num_classes, activation='softmax')))

        self.model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def __call__(self, *args, **kwargs):
        return self.model


def to_categorical(input_data, max_len, vocab_size=3):
    """
    Non-optimal conversion to categorical features
    """
    output_data = np.zeros((len(input_data), max_len, vocab_size), dtype="float32")

    for i, seqs in enumerate(input_data):
        for j, seq in enumerate(seqs):
            if j < max_len:
                output_data[i][j][seq] = 1.
            else:
                break

    return output_data


def unbpe(tokens, labels):
    words = []
    word_labels = []
    for i in range(len(tokens)):
        token = tokens[i]
        label = labels[i]
        if not token.startswith("▁") and words:
            words[-1] += token
            word_labels[-1] = label
        else:
            words.append(token)
            word_labels.append(label)
    return " ".join(word_labels)
