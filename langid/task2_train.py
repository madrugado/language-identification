import argparse

from utils import ClfModel, LangidVariantsGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--batch-size", type=int, default=256)
    parser.add_argument('-s', "--seq-len", type=int, default=256)
    parser.add_argument('-d', "--emb-dim", type=int, default=256)
    parser.add_argument('-e', "--epochs", type=int, default=5)
    parser.add_argument('-v', "--vocab-size", type=int, default=32000)

    args = parser.parse_args()
    args.classes = 2

    model = ClfModel(args).model

    train_gen = LangidVariantsGenerator(args.batch_size, args.seq_len)
    valid_gen = LangidVariantsGenerator(args.batch_size, args.seq_len, 'valid')
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=args.epochs)
    model.save('task2.h5')
