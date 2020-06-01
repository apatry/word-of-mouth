""" Models to do natural language generation.
"""

import os.path
import tensorflow as tf  # type: ignore

from abc import ABC, abstractmethod
from wordofmouth.nlg.alphabet import Alphabet
from typing import List, Optional


class ModelTrainer(ABC):
    @abstractmethod
    def loss(self, reference, output):
        """
        Loss function to use while training the model.

        :param reference: Labels in the reference.
        :param output: Output from the model.
        """
        raise NotImplementedError

    @abstractmethod
    def build(self, alphabet: Alphabet, batch_size: int):
        """Build a model generating text for a given alphabet.

        :param alphabet: Alphabet of the RNN's input and output.
        :param batch_size: Number of examples in a training batch.

        :return: A model ready to be trained or initialized.
        """
        raise NotImplementedError


class GRUModelTrainer(ModelTrainer):
    """Trainer for a natural language generation model based on Gated
    Recurrent Units.
    """

    def __init__(self, *, embedding_dim=256, rnn_units=1024):
        """
        :param embedding_dim: Size of character embeddings.
        :param rnn_unit: Number of hidden units in our RNN.
        """
        self._embedding_dim = embedding_dim
        self._rnn_units = rnn_units

    def loss(self, y, x):
        return tf.losses.sparse_categorical_crossentropy(y, x, from_logits=True)

    def build(self, alphabet, batch_size):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    len(alphabet),
                    self._embedding_dim,
                    batch_size=batch_size,
                    mask_zero=True,
                ),
                tf.keras.layers.GRU(
                    self._rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer="glorot_uniform",
                ),
                tf.keras.layers.Dense(len(alphabet)),
            ]
        )

        return model


def _create_dataset(
    alphabet: Alphabet, sentences: List[str], textwidth: int
) -> tf.data.Dataset:
    """Create a dataset for a text generation RNN.

    :param alphabet: The alphabet to train the RNN on.
    :param sentences: Sentences from which to learn the RNN.
    :param textwidth: Maximum number of characters in a given
        sentence.  Don't set it to a higher value than you need to
        avoid wasting memory while training.

    :return: A dataset where the input is all but the last character
             of each sentence and the output all but the first
             character.  This way the RNN can learn to predict the
             next character in a sentence.
    """
    encoded_sentences = [alphabet.encode(s, textwidth=textwidth) for s in sentences]
    inputs = [x[:-1] for x in encoded_sentences]
    outputs = [x[1:] for x in encoded_sentences]

    return tf.data.Dataset.from_tensor_slices((inputs, outputs))


def train_and_save(
    trainer: ModelTrainer,
    alphabet: Alphabet,
    sentences: List[str],
    model_filepath: str,
    *,
    batch_size=64,
    textwidth=32,
    checkpoint_dir: Optional[str] = None,
    epochs=8,
) -> None:
    """Train and save a model.

    The saved model operate on single instances instead of batches.
    Saved model can be loaded using `tf.keras.model.load_model`.

    :param trainer: Model trainer to run.
    :param alphabet: Input and output alphabet for the model.
    :param dataset: Dataset to train the model on.
    :param model_filepath: File where the trained model is saved.
    :param batch_size: Batch size to use during training.
    :param checkpoint_dir: Directory where snapshots are saved after
        each epoch.  Snapshot are not saved when it is none.
    :param epochs: Number of epochs to run the training for.
    """
    model = trainer.build(alphabet, batch_size)
    model.summary()
    model.compile("adam", loss=trainer.loss)

    callbacks = []
    if checkpoint_dir:
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True
        )
        callbacks.append(checkpoint_callback)

    # fit the model on the dataset
    dataset = (
        _create_dataset(alphabet, sentences, textwidth)
        .shuffle(len(sentences), reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
    )
    model.fit(dataset, epochs=epochs, callbacks=callbacks)
    model.save_weights(model_filepath)
    model2 = trainer.build(alphabet, 1)
    model2.load_weights(model_filepath)
    model2.build()
    model2.save_weights(model_filepath)


def _generate(
    model, prefix: List[int], eos: int, max_length: int, gru_states: Optional[List]
):
    """
    Generate a sequence from a recurrent model.

    :param model: Model to generate the sequence with.
    :param prefix: Prefix to continue generating with the model.
    :param eos: Id to identify the end of a sequence (should be 0 most
        of the time).
    :param max_length: Maximum lenght of the sequence to generate.
    :param gru_states: State of the GRU cell as predictions are made.
    """
    output_values = []
    predictions = None
    model.reset_states()

    for i in range(len(prefix) + max_length):
        if i < len(prefix):
            x_i = prefix[i]
        else:
            assert predictions is not None
            x_i = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        if x_i == eos:
            break

        input_values = tf.expand_dims([x_i], 0)
        predictions = model.predict(input_values, batch_size=1)
        predictions = tf.squeeze(predictions, 0)

        if gru_states is not None:
            # TODO we should have a callback here instead
            gru_states.append(model.get_layer("gru").states[0][0, :].numpy())

        output_values.append(x_i)

    return output_values


def generate_text(
    model, alphabet: Alphabet, start_string: str, *, max_length=16, gru_states=None
) -> str:
    """
    Generate text using a NLG model.

    :param model: Model to generate text with.
    :param alphabet: Output alphabet of the model.
    :param max_length: The maximum length of the text to generate.

    :return: Generated text.
    """
    start = alphabet.encode(start_string)
    encoded_text = _generate(model, start, alphabet.padding_id, max_length, gru_states)
    return alphabet.decode(encoded_text)
