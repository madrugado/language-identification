import argparse

from task3_utils import CodeSwitchingGenerator, Seq2SeqModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--batch-size", type=int, default=256)
    parser.add_argument('-s', "--seq-len", type=int, default=256)
    parser.add_argument('-d', "--emb-dim", type=int, default=256)
    parser.add_argument('-e', "--epochs", type=int, default=5)
    parser.add_argument('-v', "--vocab-size", type=int, default=32000)
    parser.add_argument('--hidden-size', type=int, default=128)

    args = parser.parse_args()
    args.num_classes = 3

    model = Seq2SeqModel(args).model

    train_gen = CodeSwitchingGenerator(args.batch_size, args.seq_len)
    valid_gen = CodeSwitchingGenerator(args.batch_size, args.seq_len, 'dev')
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=args.epochs)
    model.save('task3.h5')
