""" A library to create a gazetteer from wikidata. """

import io
import requests

from typing import TextIO

SPARQL_URL = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"


def download_as_csv_file(query: str, fd: TextIO) -> None:
    """Download the result of a Sparql query to Wikidata into a csv file.

  :param query: A Sparql query.
  :param fd: A file like object where to save the results.
  """
    response = requests.get(
        SPARQL_URL,
        params={"query": query},
        headers={"Accept": "text/csv; charset=utf8"},
    )

    for chunk in response.iter_content(chunk_size=128):
        fd.write(chunk)


def download_bands_gazetteer(fd: TextIO) -> None:
    """Download a gazetteer of music bands from Wikidata.

  The resulting gazetteer is a CSV file with two columns:

  - entity: The wikidata entity for the band;
  - label: The name of the band.

  :param fd: A file like object where the csv file is stored.
  """
    query = """
    SELECT DISTINCT ?entity ?label
    WHERE
    {
      ?entity wdt:P31/wdt:P279* wd:Q2088357.
      ?entity rdfs:label ?label.
      FILTER (LANG(?label) = 'en')
    }
  """
    download_as_csv_file(query, fd)


def dowload_musicians_gazetteer(fd: TextIO) -> None:
    """Download a gazetteer of musicians from Wikidata.

  The resulting gazetteer is a CSV file with two columns:

  - entity: The wikidata entity for the musician;
  - label: The name of the musician.

  :param fd: A file like object where the csv file is stored.
  """
    query = """
    SELECT DISTINCT ?entity ?label
    WHERE
    {
      ?entity wdt:P31 wd:Q5;
                wdt:P106/wdt:P279* wd:Q639669.

      ?entity rdfs:label ?label.
      FILTER (LANG(?label) = 'en')
    }
  """
    download_as_csv_file(query, fd)
