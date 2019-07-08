from keras.utils import Sequence, to_categorical
import sentencepiece as spm
from keras.preprocessing.sequence import pad_sequences

from keras import Model
from keras.constraints import MaxNorm
from keras.layers import Dense, Activation, Embedding, Input, concatenate, LSTM, GlobalMaxPooling1D, Conv1D, \
    Bidirectional, Masking
from tqdm import tqdm

lang_code = {'en': 0, 'es': 1, 'pt': 2}
lang_code_variants = {'br': 0, 'pt': 1}


class LangidGenerator(Sequence):
    def __init__(self, batch_size, seq_len, mode='train'):
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("langid.all.model")

        with open("langid-data/task1/data.es.tok") as f:
            self.data_es = f.readlines()
        with open("langid-data/task1/data.en.tok") as f:
            self.data_en = f.readlines()
        with open("langid-data/task1/data.pt.tok") as f:
            self.data_pt = f.readlines()

        self.size = len(self.data_en)  # It is the smallest one

        valid_size = self.size // 10
        if mode == 'train':
            self.data_es = self.data_es[valid_size:]
            self.data_en = self.data_en[valid_size:]
            self.data_pt = self.data_pt[valid_size:]
            self.size -= valid_size
        else:
            self.data_es = self.data_es[:valid_size]
            self.data_en = self.data_en[:valid_size]
            self.data_pt = self.data_pt[:valid_size]
            self.size = valid_size

        self.normalize_data()

    def normalize_data(self):
        for idx in tqdm(range(len(self.data_es))):
            self.data_es[idx] = [self.sp.piece_to_id(token) for token in self.data_es[idx].strip().split()]
        for idx in tqdm(range(len(self.data_en))):
            self.data_en[idx] = [self.sp.piece_to_id(token) for token in self.data_en[idx].strip().split()]
        for idx in tqdm(range(len(self.data_pt))):
            self.data_pt[idx] = [self.sp.piece_to_id(token) for token in self.data_pt[idx].strip().split()]

    def __getitem__(self, index):
        en_batch_size = self.batch_size // 3
        pt_batch_size = self.batch_size // 3
        es_batch_size = self.batch_size - 2 * (self.batch_size // 3)
        batch_en = self.data_en[index * en_batch_size: (index + 1) * en_batch_size]
        batch_es = self.data_es[index * es_batch_size: (index + 1) * es_batch_size]
        batch_pt = self.data_pt[index * pt_batch_size: (index + 1) * pt_batch_size]

        return pad_sequences(batch_en + batch_es + batch_pt, self.seq_len), \
               to_categorical([lang_code['en']] * en_batch_size + [lang_code['es']] * es_batch_size
                              + [lang_code['pt']] * pt_batch_size, 3)

    def __len__(self):
        return self.size // (self.batch_size // 3 + 2)  # Quick and dirty size estimation


class LangidVariantsGenerator(Sequence):
    def __init__(self, batch_size, seq_len, mode='train'):
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("langid.brpt.model")

        with open("langid-data/task2/data.pt-br.tok") as f:
            self.data_br = f.readlines()
        with open("langid-data/task2/data.pt-pt.tok") as f:
            self.data_pt = f.readlines()

        self.size = len(self.data_br)  # It is the smallest one

        valid_size = self.size // 10
        if mode == 'train':
            self.data_br = self.data_br[valid_size:]
            self.data_pt = self.data_pt[valid_size:]
            self.size -= valid_size
        else:
            self.data_br = self.data_br[:valid_size]
            self.data_pt = self.data_pt[:valid_size]
            self.size = valid_size

        self.normalize_data()

    def normalize_data(self):
        for idx in tqdm(range(len(self.data_br))):
            self.data_br[idx] = [self.sp.piece_to_id(token) for token in self.data_br[idx].strip().split()]
        for idx in tqdm(range(len(self.data_pt))):
            self.data_pt[idx] = [self.sp.piece_to_id(token) for token in self.data_pt[idx].strip().split()]

    def __getitem__(self, index):
        pt_batch_size = self.batch_size // 2
        br_batch_size = self.batch_size - (self.batch_size // 2)
        batch_br = self.data_br[index * br_batch_size: (index + 1) * br_batch_size]
        batch_pt = self.data_pt[index * pt_batch_size: (index + 1) * pt_batch_size]

        return pad_sequences(batch_br + batch_pt, self.seq_len), \
               to_categorical([lang_code_variants['br']] * br_batch_size
                              + [lang_code['pt']] * pt_batch_size, 2)

    def __len__(self):
        return self.size // (self.batch_size // 2)  # Quick and dirty size estimation


class ClfModel:
    def __init__(self, args):
        # Model definition
        inp = Input((args.seq_len,))
        word_emb = Embedding(args.vocab_size, args.emb_dim,
                             mask_zero=False, name='word_emb',
                             embeddings_constraint=MaxNorm(10))

        embedded = word_emb(inp)
        y_s = Conv1D(args.emb_dim, 3)(embedded)
        y_s = Conv1D(args.emb_dim, 3)(y_s)
        y_s = GlobalMaxPooling1D()(y_s)
        classes = Dense(args.classes, activation='softmax')(y_s)
        self.model = Model(inputs=[inp], outputs=[classes])

        self.model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def __call__(self, *args, **kwargs):
        return self.model


def tokenize_corpora(lang_list, sp, task=1):
    for lang in lang_list:
        with open("langid-data/task{}/data.{}".format(task, lang)) as f_in, \
                open("langid-data/task{}/data.{}.tok".format(task, lang), "wt") as f_out:
            for line in tqdm(f_in):
                f_out.write(" ".join(sp.encode_as_pieces(line)) + "\n")
