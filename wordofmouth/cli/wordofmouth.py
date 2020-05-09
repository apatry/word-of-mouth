"""
A program to download gazetteer from wikidata.
"""

import sys

from wordofmouth import bandaid
from wordofmouth.cli.commands import Command, main
from wordofmouth.gazetteer import download_bands_gazetteer
from wordofmouth.nlg.models import GRUModelTrainer

from typing import Optional, List


class BandaidTrainCommand(Command):
    """
    A `Command` to train a band name generation model.
    """

    def configure(self, parser):
        parser.add_argument(
            "--training-dataset",
            metavar="FILE",
            required=True,
            help="Input file containing the training dataset.",
        )
        parser.add_argument(
            "--codec",
            metavar="FILE",
            required=True,
            help="Output file where the character codec should be saved.",
        )
        parser.add_argument(
            "--model",
            metavar="FILE",
            required=True,
            help="Output prefix for files where the architecture configuration is stored.",
        )
        parser.add_argument(
            "--weights",
            metavar="FILE_PREFIX",
            required=True,
            help="Output prefix for files where the model weights are stored.",
        )

    def execute(self, arguments):
        bands = bandaid.create_dataset(arguments.training_dataset)
        trainer = GRUModelTrainer(embedding_dim=64, rnn_units=512)
        bandaid.train(
            trainer, bands, arguments.model, arguments.weights, arguments.codec
        )

    @property
    def name(self):
        return "bandaid-train"

    @property
    def help(self):
        return "Train a new band name generator model."


class BandaidGenerateCommand(Command):
    """
    A `Command` to generate music band names from a trained model and
    a prefix.
    """

    def configure(self, parser):
        parser.add_argument(
            "--codec",
            metavar="FILE",
            required=True,
            help="File where the character codec is saved.",
        )
        parser.add_argument(
            "--model",
            metavar="FILE",
            required=True,
            help="Prefix for the files where the architecture configuration is stored.",
        )
        parser.add_argument(
            "--weights",
            metavar="FILE_PREFIX",
            required=True,
            help="Prefix for the files where the model weights are stored.",
        )

        parser.add_argument(
            "prefixes",
            metavar="STR",
            nargs="+",
            help="Band name prefix to complete via text generation.",
        )

    def execute(self, arguments):
        recs = bandaid.generate(
            arguments.model, arguments.weights, arguments.codec, arguments.prefixes
        )
        print("\n".join(recs))

    @property
    def name(self):
        return "bandaid"

    @property
    def help(self):
        return "Generate a new band name."


class DownloadBandsGazetteerCommand(Command):
    """
    A `Command` to download a dataset of music bands from wikidata.
    """

    def configure(self, parser) -> None:
        parser.add_argument(
            "output", metavar="CSVFILE", help="File where the dataset is saved"
        )

    def execute(self, arguments) -> None:
        with open(arguments.output, "wb") as f:
            download_bands_gazetteer(f)

    @property
    def name(self):
        return "download-bands-dataset"

    @property
    def help(self):
        return "Download a dataset of music bands from wikidata."


def run(argv: Optional[List[str]] = None) -> None:
    commands = [
        BandaidGenerateCommand(),
        BandaidTrainCommand(),
        DownloadBandsGazetteerCommand(),
    ]
    main("wordofmouth", commands, argv or sys.argv)


if __name__ == "__main__":
    run()
