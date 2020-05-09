""" Models to do natural language generation.
"""

import collections
import numpy as np  # type: ignore
import os.path
import tempfile
import tensorflow as tf  # type: ignore
import uuid

from abc import ABC, abstractmethod
from wordofmouth.nlg.alphabet import Alphabet
from typing import List, NewType, Optional


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


# text
# score
# next (can be empty for terminal nodes)


def generate_graph(
    model,
    alphabet: Alphabet,
    start_string: str,
    *,
    max_length: int = 16,
    fanout: int = 2,
) -> str:
    """
    Generate a search graph of possible completion for a prefix.

    The graph is in dot format.

    :param model: Model to generate the graph with.
    :param alphabet: Output alphabet of the model.
    :param max_length: The maximum length of the text to generate.
    :param fanout: Number of hypothesis per node to explore.

    :return: The search space explored as a graph.
    """
    pass


SearchNode = collections.namedtuple(
    "SearchNode", ["previous", "score", "sequence", "length", "terminal"]
)


def sample_next():
    pass


class SearchNode:
    def __init__(
        self, score: float, sequence: List[int], previous: Optional[SearchNode] = None
    ):
        self._id = uuid.uuid4()
        self._previous = previous
        self._score = score
        self._sequence = sequence

    @property
    def id(self) -> uuid.UUID:
        return self._id

    @property
    def previous(self) -> Optional[SearchNode]:
        return self._previous

    @property
    def score(self) -> float:
        return self._score

    @property
    def sequence(self) -> List[int]:
        return self._sequence

    @property
    def path(self) -> List[int]:
        value = self.sequence
        node = self
        while node.previous:
            node = node.previous
            value = node.sequence + value
        return value

    @property
    def path_score(self) -> float:
        value = self.score
        x = 1
        node = self
        while node.previous:
            node = node.previous
            value += node.score
            x += 1
        return value / x

    @property
    def nodes(self) -> List[uuid.UUID]:
        value = [self._id]
        node = self
        while node.previous:
            node = node.previous
            value = [node.id] + value
        return value

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def terminal(self) -> bool:
        return not self.sequence


def _generate_graph(
    model, prefix: List[int], eos: int, max_depth: int, fanout: int, beam: int = 5
):
    input_values = tf.expand_dims(prefix, 0)
    output_values = prefix + []

    completed_nodes = []
    planned_nodes = [SearchNode(1, prefix)]
    current_nodes = []

    for i in range(max_depth):
        planned_nodes = sorted(planned_nodes, key=lambda x: x.path_score)
        # only process top-k
        completed_nodes.extend(planned_nodes[:-beam])
        current_nodes = planned_nodes[-beam:]

        # reset planned nodes to a clean slate
        planned_nodes = []

        for current_node in current_nodes:
            input_values = tf.expand_dims(current_node.path, 0)
            model.reset_states()
            predictions = model.predict(input_values, batch_size=1)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            scores = predictions[-1, :]
            probs = np.exp(scores) / sum(np.exp(scores))
            predicted_ids = np.random.choice(len(probs), fanout, replace=True, p=probs)

            for predicted_id in set(predicted_ids):
                score = predictions[-1, predicted_id].numpy()
                terminal = predicted_id == eos

                if terminal:
                    next_node = SearchNode(score, [], current_node)
                    completed_nodes.append(next_node)
                else:
                    next_node = SearchNode(score, [predicted_id], current_node)
                    planned_nodes.append(next_node)

    completed_nodes.extend(planned_nodes)
    return sorted(
        completed_nodes,
        key=lambda x: (x.terminal, x.path_score, x.depth, x.score),
        reverse=True,
    )


def print_hypothesis(alphabet, hypothesis):
    for h in hypothesis:
        print(f"{alphabet.decode(h.path)}\t{h.score}\t{h.path_score}\t{h.terminal}")


def print_graph(alphabet, hypothesis):
    visited = set()
    edges = []
    nodes = []
    agenda = list(hypothesis)
    while agenda:
        h = agenda.pop()
        id = str(h.id).replace("-", "_")
        if id in visited:
            continue
        nodes.append(
            f'node_{id}[label="{alphabet.decode(h.path)}" {"fillcolor=lightblue style=filled" if h.terminal else ""}]'
        )
        if h.previous:
            previous_id = str(h.previous.id).replace("-", "_")
            edges.append(
                f'node_{previous_id} -> node_{id} [label="{alphabet.decode(h.sequence)} ({h.score:0.02})"]'
            )
            agenda.append(h.previous)
        visited.add(id)

    graph = "digraph {\n\t"
    graph += "\n\t".join(nodes) + "\n\t"
    graph += "\n\t".join(edges) + "\n"
    graph += "}\n"

    return graph
