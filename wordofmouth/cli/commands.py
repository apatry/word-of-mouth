"""
Utility to help at the creation of commands.
"""

import argparse

from abc import ABC, abstractmethod
from typing import Sequence


class Command(ABC):
    """
    Defines a command to be run from a Command Line Interface.

    The `name` property is the name of the command and the `help` a
    short description of it.

    A program using commands should be wired like this:

    ```
    import sys

    commands = [MyCommand1(), MyCommand2(), ...]
    main("PROG", commands, sys.args)
    ```
    """

    name: str
    help: str

    @abstractmethod
    def configure(self, parser) -> None:
        """
        Configure the command line argument parser.

        See the documentation of `argparse` to learn more about how to
        configure the parser.

        :param parser: `argparse` parser to configure.
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, arguments) -> None:
        """
        Execute the command.

        :param arguments: The command arguments that were parsed by
            `argparse`.
        """
        raise NotImplementedError


def main(prog: str, commands: Sequence[Command], args: Sequence[str]) -> None:
    """
    Run subcommands to train and generate music band name generation
    models.
    """
    parser = argparse.ArgumentParser(prog=prog)
    subparsers = parser.add_subparsers(dest="command_name")

    for command in commands:
        subparser = subparsers.add_parser(command.name, help=command.help)
        command.configure(subparser)

    parsed_arguments = parser.parse_args(args)

    for command in commands:
        if command.name == parsed_arguments.command_name:
            command.execute(parsed_arguments)
