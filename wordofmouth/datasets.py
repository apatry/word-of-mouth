from typing import Optional

from peewee import chunked
from wordofmouth.database import MusicBand
from wordofmouth.etl.sparql import query_wikidata


def etl_music_bands(db, *, chunk_size: int = 512, limit: Optional[int] = None) -> int:
    """
    ETL music bands into our databases.

    :param db: The database in which to ETL the bands.
    :param chunk_size: The number of records in each insert batch.
    :param limit: The maximum number of records to fetch from
        wikidata.  This is included for debugging purpose.

    :return: The number of records that were ETLed.
    """

    query = """
    SELECT DISTINCT ?wikidata_id ?name
    WHERE
    {
      ?wikidata_id wdt:P31/wdt:P279* wd:Q2088357.
      ?wikidata_id rdfs:label ?name.
      FILTER (LANG(?name) = 'en')
    }
    """

    if limit:
        query = f"{query}\nLIMIT {limit}"

    bands = query_wikidata(query)
    with db.atomic():
        for batch in chunked(bands, chunk_size):
            MusicBand.insert_many(batch).execute()

    return len(bands)
