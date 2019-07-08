import logging
import numpy as np
import sentencepiece as spm
import argparse
from gensim.models import Word2Vec

from tqdm import tqdm

if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", type=str, default='enes')
    args = parser.parse_args()

    with open("../langid/langid-data/task1/data.en") as f_in, \
            open("../langid/langid-data/task1/data.{}".format(args.langs), "wt") as f_out:
        for line in f_in:
            f_out.write(line + "\n")
    with open("../langid/langid-data/task1/data.es") as f_in, \
            open("../langid/langid-data/task1/data.{}".format(args.langs), "at") as f_out:
        for line in f_in:
            f_out.write(line + "\n")

    spm.SentencePieceTrainer.Train('--input=../langid/langid-data/task1/data.{} --model_prefix=langid.{} '
                                   '--vocab_size=32000'
                                   ' --bos_id=-1 --eos_id=-1 --pad_id=0 --unk_id=1'.format(args.langs, args.langs))

    sp = spm.SentencePieceProcessor()
    sp.Load("langid.{}.model".format(args.langs))

    with open("../langid/langid-data/task1/data.{}".format(args.langs)) as f_in, \
            open("../langid/langid-data/task1/data.{}.tok".format(args.langs), "wt") as f_out:
        for line in tqdm(f_in):
            f_out.write(" ".join(sp.encode_as_pieces(line)) + "\n")

    sentences = []
    with open("../langid/langid-data/task1/data.{}.tok".format(args.langs)) as f:
        for line in f:
            sentences.append(line.split())

    model = Word2Vec(sentences, size=300, window=5, min_count=0, workers=20, iter=10)
    model.wv.save_word2vec_format("task3_w2v", binary=True)

    matrix = np.array([sp.IdToPiece(i) for i in range(1, 32000)])
    matrix = np.array([np.zeros(300)] + [model.wv[piece] if piece in model.wv.vocab else np.zeros(300)
                                         for piece in matrix])
    np.save("embeddings.npy", matrix)
