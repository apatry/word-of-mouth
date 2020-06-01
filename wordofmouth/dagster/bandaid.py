""" Pipelines to train a bandaid model. """

from wordofmouth import bandaid
from wordofmouth.nlg.models import GRUModelTrainer
from wordofmouth.gazetteer import download_bands_gazetteer

from dagster import (  # type: ignore
    lambda_solid,
    pipeline,
    solid,
    Field,
    Int,
    Output,
    OutputDefinition,
    Path,
)

from typing import List


@lambda_solid
def download_dataset(csv_path: str) -> Path:
    """
    Download a dataset of band names.

    :param context: Dagster context.
    :param csv_path: Path where to save the dataset.

    :return: The path where the dataset is saved.
    """
    with open(csv_path, "wb") as output:
        download_bands_gazetteer(output)
    return csv_path


@solid(
    config={
        "batch_size": Field(Int, is_required=False, default_value=256),
        "epochs": Field(Int, is_required=False, default_value=8),
        "embedding_dim": Field(Int, is_required=False, default_value=64),
        "rnn_units": Field(Int, is_required=False, default_value=512),
    },
    output_defs=[
        OutputDefinition(name="model", dagster_type=Path),
        OutputDefinition(name="weights", dagster_type=Path),
        OutputDefinition(name="codec", dagster_type=Path),
    ],
)
def train(context, dataset: str, output_prefix: str):
    """
    Train a model to generate band names.

    :param context: Dagster context.
    :param dataset: The path to the training dataset.
    :param output_prefix: Prefix of the output files for the trained
        model.
    """
    bands = bandaid.create_dataset(dataset)
    trainer = GRUModelTrainer(
        embedding_dim=context.solid_config["embedding_dim"],
        rnn_units=context.solid_config["rnn_units"],
    )
    model = f"{output_prefix}-model.bin"
    weights = f"{output_prefix}-weights.bin"
    codec = f"{output_prefix}-codec.bin"
    bandaid.train(
        trainer,
        bands,
        model,
        weights,
        codec,
        batch_size=context.solid_config["batch_size"],
        epochs=context.solid_config["epochs"],
    )
    yield Output(model, "model")
    yield Output(weights, "weights")
    yield Output(codec, "codec")


@lambda_solid
def generate(
    model: Path, weights: Path, codec: Path, prefixes: List[str], band_names: Path
):
    """
    Generate the name of a band.

    :param model: The model file.
    :param weights: The model weights file.
    :param codec: The codec file.
    :param prefixes: The band name prefixes.
    :param band_names: File where to save the generated band names.

    :return: One band name per prefix.
    """
    bands = bandaid.generate(model, weights, codec, prefixes)
    with open(band_names, "w") as f:
        for (prefix, band) in zip(prefixes, bands):
            f.write(f"{prefix}\t{band}\n")


@pipeline
def training_pipeline():
    """
    Train a bandname generation model.
    """
    dataset = download_dataset()
    (model, weights, codec) = train(dataset)
    generate(model, weights, codec)
