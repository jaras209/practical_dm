import argparse
from dataset import Dataset
import tensorflow as tf
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--emb_dim", default=64, type=int, help="Character embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_file", default="cz_sk_train.csv", type=str, help="Name of the train data file.")
parser.add_argument("--val_file", default="cz_sk_val.csv", type=str, help="Name of the val data file.")
parser.add_argument("--save_folder", default="save_folder", type=str,
                    help="Name of the folder where to save the model or where to load it from")
parser.add_argument('--train', dest='train_model', action='store_true')
parser.add_argument('--test', dest='train_model', action='store_false')
parser.set_defaults(train_model=True)
args = parser.parse_args([] if "__file__" not in globals() else None)


def run_model(dataset: Dataset, emb_dim, rnn_dim, batch_size, epochs):
    utterance_input = tf.keras.layers.Input(shape=(None,))
    utterance_embed = tf.keras.layers.Embedding(input_dim=dataset.num_words, output_dim=emb_dim, mask_zero=True)(
        utterance_input)
    utterance_rnn = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(units=rnn_dim, return_sequences=False), merge_mode='sum')(utterance_embed)

    history_input = tf.keras.layers.Input(shape=(None,))
    history_embed = tf.keras.layers.Embedding(input_dim=dataset.num_words, output_dim=emb_dim, mask_zero=True)(
        history_input)
    history_rnn = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(units=rnn_dim, return_sequences=False), merge_mode='sum')(history_embed)

    concat = tf.keras.layers.Concatenate()([utterance_rnn, history_rnn])
    actions = tf.keras.layers.Dense(units=dataset.num_actions, activation=tf.nn.sigmoid)(concat)

    model = tf.keras.Model(inputs=[utterance_input, history_input], outputs=actions)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])
    model.summary()

    training_history = model.fit([dataset.train_utterances, dataset.train_history], dataset.train_actions,
                                 batch_size=batch_size, epochs=epochs,
                                 validation_data=([dataset.val_utterances, dataset.val_history], dataset.val_actions),
                                 )
    model.save("model.h5")


def main(dataset: Dataset):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    run_model(dataset, emb_dim=args.emb_dim, rnn_dim=args.rnn_dim, batch_size=args.batch_size, epochs=args.epochs)
