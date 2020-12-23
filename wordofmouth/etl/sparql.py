"""
ETL data from wikidata using sparql queries.
"""
import csv
from contextlib import closing
from typing import Dict, List

import requests

# Sparql end-point for wikidata.
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"


def query_wikidata(query: str) -> List[Dict]:
    """Run a sparql query on wikidata.

    We load all the results in memory.  This function is not
    appropriate if the data is too large to fit in memory.

    :param query: A Sparql query.

    :return: A list of dictionaries where keys correspond to the
             columns returned by the query.
    """
    response = requests.get(
        WIKIDATA_SPARQL_URL,
        params={"query": query},
        headers={"Accept": "text/csv; charset=utf-8"},
        stream=True,
    )

    with closing(response):
        lines = (line.decode("utf-8") for line in response.iter_lines())
        reader = csv.DictReader(lines)
        return list(reader)
