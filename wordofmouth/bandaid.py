"""
A music band name generator.

I had a bass for the last 25 years but almost never used it and have
always been terrible at it.  This module will help me generate bands
name that are on part with my musical talent.

In all honesty, it is just an excuse to perform text generation using
Recurrent Neural Networks (RNN) on a dataset I had at hand.
"""

import collections
import csv
import pickle

from typing import List
from wordofmouth.nlg.alphabet import Alphabet
from wordofmouth.nlg.models import ModelTrainer, generate_text, train_and_save


def create_dataset(filename: str) -> List[str]:
    with open(filename) as f:
        lines = csv.reader(f)
        return [
            name.lower()
            for entity, name in lines
            if all(ord(c) < 128 for c in name) and len(name) <= 32
        ][1:]


def train(
    trainer: ModelTrainer,
    bands: List[str],
    trainer_filepath: str,
    model_filepath: str,
    alphabet_filepath: str,
    batch_size: int = 256,
    epochs: int = 8,
) -> None:
    alphabet = Alphabet(bands)

    with open(trainer_filepath, "bw") as f:
        pickle.dump(trainer, f)

    with open(alphabet_filepath, "bw") as f:
        pickle.dump(alphabet, f)

    train_and_save(
        trainer,
        alphabet,
        bands,
        model_filepath,
        batch_size=min(batch_size, len(bands)),
        epochs=epochs,
    )


BandaidModel = collections.namedtuple("BandaidModel", ["alphabet", "model"])


def load_model(trainer_filepath: str, model_filepath: str, alphabet_filepath: str):

    with open(trainer_filepath, "rb") as f:
        trainer = pickle.load(f)

    with open(alphabet_filepath, "rb") as f:
        alphabet = pickle.load(f)

    model = trainer.build(alphabet, 1)
    model.load_weights(model_filepath)

    return BandaidModel(alphabet, model)


def generate(
    trainer_filepath: str,
    model_filepath: str,
    alphabet_filepath: str,
    prefixes: List[str],
):
    with open(trainer_filepath, "rb") as f:
        trainer = pickle.load(f)

    with open(alphabet_filepath, "rb") as f:
        alphabet = pickle.load(f)

    model = trainer.build(alphabet, 1)
    model.load_weights(model_filepath)

    return [
        generate_text(model, alphabet, prefix or alphabet.sample())
        for prefix in prefixes
    ]


def format_bands_embeddings(
    model: BandaidModel,
    bandnames: List[str],
    paths: List[str],
    metadata_tsv: str,
    tensors_tsv: str,
) -> None:
    """
    Create data that can be used in Embedding Projector to visualize
    band names and band name prefixes embeddings.

    :param model: The model creating the embeddings.
    :param bandnames: The list of band names to project.
    :param paths: The prefix to project.
    :param metadata_tsv: The file where to write the metadata.
    :param tensors_tsv: The file where to write the tensors.
    """
    with open(metadata_tsv, "w", encoding="utf-8") as m, open(
        tensors_tsv, "w", encoding="utf-8"
    ) as t:
        m.write("name\tclass\n")
        for band in bandnames:
            m.write(f"{band}\ttraining\n")

            generate_text(model.model, model.alphabet, band + "<", max_length=0)
            embedding = model.model.get_layer("gru").states[0][0, :].numpy()
            t.write("\t".join(str(x) for x in embedding) + "\n")

        for path in paths:
            embeddings: List[List[float]] = []
            generate_text(
                model.model,
                model.alphabet,
                f"{path}<",
                max_length=0,
                gru_states=embeddings,
            )
            for (i, embedding) in enumerate(embeddings):
                m.write(f"{path[:i+1]}\t{path}\n")
                t.write("\t".join([str(x) for x in embedding]) + "\n")
