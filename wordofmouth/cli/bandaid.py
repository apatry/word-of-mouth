"""
A program to generate new music band names.
"""

import sys

from wordofmouth.bandaid import create_dataset, generate_text, train
from wordofmouth.cli.commands import Command, main

class TrainCommand(Command):
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
            metavar="FILE_PREFIX",
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
        bands = read_band_dataset(arguments.training_dataset)
        train(GRUTrainer(), arguments.model, arguments.weights, arguments.codec)

    @property
    def name(self):
        return "train"

    @property
    def help(self):
        return "Train a new band name generator model."


class GenerateCommand(Command):
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
            metavar="FILE_PREFIX",
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
        for prefix in arguments.prefixes:
            print(generate(arguments.model, arguments.weights, arguments.codec, prefix))

    @property
    def name(self):
        return "generate"

    @property
    def help(self):
        return "Generate a new band name."


def run() -> None:
    commands = [TrainCommand(), GenerateCommand()]
    main("bandaid", commands, sys.argv)

if __name__ == "__main__":
    run()
