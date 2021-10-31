import argparse
from dataset import Dataset
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--emb_dim", default=64, type=int, help="Character embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--rnn_dim", default=128, type=int, help="RNN cell dimension.")
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


def f1_metrics(y_true, y_pred):  # shapes (batch, actions)
    predLabels = K.argmax(y_pred, axis=-1)
    y_pred = tf.cast(K.one_hot(predLabels, 247), tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    ground_positives = tf.reduce_sum(y_true, axis=0) + K.epsilon()  # = TP + FN
    pred_positives = tf.reduce_sum(y_pred, axis=0) + K.epsilon()  # = TP + FP
    true_positives = tf.reduce_sum(y_true * y_pred, axis=0) + K.epsilon()  # = TP
    # all with shape (4,)

    precision = true_positives / pred_positives
    recall = true_positives / ground_positives
    # both = 1 if ground_positives == 0 or pred_positives == 0
    # shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    # still with shape (4,)

    weighted_f1 = f1 * ground_positives / tf.reduce_sum(ground_positives)
    weighted_f1 = tf.reduce_sum(weighted_f1)

    return weighted_f1


def f1_loss(y_true, y_pred):  # shapes (batch, actions)
    y_true = tf.cast(y_true, tf.float32)
    ground_positives = tf.reduce_sum(y_true, axis=0) + K.epsilon()  # = TP + FN
    pred_positives = tf.reduce_sum(y_pred, axis=0) + K.epsilon()  # = TP + FP
    true_positives = tf.reduce_sum(y_true * y_pred, axis=0) + K.epsilon()  # = TP
    # all with shape (4,)

    precision = true_positives / pred_positives
    recall = true_positives / ground_positives
    # both = 1 if ground_positives == 0 or pred_positives == 0
    # shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    # still with shape (4,)

    weighted_f1 = f1 * ground_positives / tf.reduce_sum(ground_positives)
    weighted_f1 = tf.reduce_sum(weighted_f1)

    return 1 - weighted_f1


def run_model(dataset: Dataset, emb_dim, rnn_dim, batch_size, epochs):
    utterance_input = tf.keras.layers.Input(shape=(None,))
    utterance_embed = tf.keras.layers.Embedding(input_dim=dataset.num_words, output_dim=emb_dim, mask_zero=True)(
        utterance_input)
    utterance_rnn = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(units=rnn_dim, return_sequences=True), merge_mode='sum')(utterance_embed)

    history_input = tf.keras.layers.Input(shape=(None,))
    history_embed = tf.keras.layers.Embedding(input_dim=dataset.num_words, output_dim=emb_dim, mask_zero=True)(
        history_input)
    history_rnn = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(units=rnn_dim, return_sequences=True), merge_mode='sum')(history_embed)

    concat = tf.keras.layers.Concatenate(axis=1)([utterance_rnn, history_rnn])
    inner_rnn = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(units=rnn_dim, return_sequences=False), merge_mode='sum')(concat)

    actions = tf.keras.layers.Dense(units=dataset.num_actions, activation=tf.nn.sigmoid)(inner_rnn)

    model = tf.keras.Model(inputs=[utterance_input, history_input], outputs=actions)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=f1_loss, metrics=[f1_metrics])
    model.summary()

    training_history = model.fit([dataset.train_utterances, dataset.train_history], dataset.train_actions,
                                 batch_size=batch_size, epochs=epochs,
                                 validation_data=([dataset.val_utterances, dataset.val_history], dataset.val_actions),
                                 )
    model.save("model.h5")


def test_model(dataset: Dataset, batch_size):
    model = tf.keras.models.load_model('model.h5')
    model.summary()

    prediction = np.rint(model.predict([dataset.test_utterances, dataset.test_history], batch_size=batch_size))

    equal = np.sum(np.all(prediction == dataset.test_actions, axis=1))

    y_true = tf.convert_to_tensor(dataset.test_actions, dtype=tf.int32)

    #TODO: zavolat spis nejake evaulate nebo tak neco, co tohle pocita?
    print(equal / prediction.shape[0])
    print(f1_metrics(y_true=y_true, y_pred=prediction))


def main(dataset: Dataset, train=True):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if train:
        run_model(dataset, emb_dim=args.emb_dim, rnn_dim=args.rnn_dim, batch_size=args.batch_size, epochs=args.epochs)
    else:
        test_model(dataset, batch_size=args.batch_size)
