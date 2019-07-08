import logging

import sentencepiece as spm
import argparse

from utils import tokenize_corpora

if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", type=str, default='all')
    args = parser.parse_args()

    with open("../langid/langid-data/task1/data.en") as f_in, \
            open("../langid/langid-data/task1/data.{}".format(args.langs), "wt") as f_out:
        for line in f_in:
            f_out.write(line + "\n")
    with open("../langid/langid-data/task1/data.es") as f_in, \
            open("../langid/langid-data/task1/data.{}".format(args.langs), "at") as f_out:
        for line in f_in:
            f_out.write(line + "\n")
    with open("../langid/langid-data/task1/data.pt") as f_in, \
            open("../langid/langid-data/task1/data.{}".format(args.langs), "at") as f_out:
        for line in f_in:
            f_out.write(line + "\n")

    spm.SentencePieceTrainer.Train('--input=langid-data/task1/data.{} --model_prefix=langid.{} --vocab_size=32000'
                                   ' --bos_id=-1 --eos_id=-1 --pad_id=0 --unk_id=1'.format(args.langs, args.langs))

    sp = spm.SentencePieceProcessor()
    sp.Load("langid.{}.model".format(args.langs))
    tokenize_corpora(["en", "es", "pt"], sp)
