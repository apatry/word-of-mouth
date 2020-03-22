"""
Minimalistic crawler to create datasets from reddit.

This crawler leverage the
[pushshift api](https://github.com/pushshift/api).
"""

import requests
import time
from typing import List, Dict

# Pushshift end point serving submissions
submissions_url = "https://api.pushshift.io/reddit/search/submission"


def crawl_page(subreddit: str, last_page=None, *, page_size: int = 500):
    """
  Crawl a page of results from a given subreddit.

  :param subreddit: The subreddit to crawl.
  :param last_page: The last downloaded page.

  :return: A page or results.
  """
    if last_page is not None and not last_page:
        # the last page was empty, no need to crawl another page
        return []

    params = {
        "subreddit": subreddit,
        "size": str(page_size),
        "sort": "desc",
        "sort_type": "created_utc",
    }

    if last_page is not None:
        # resume from last search results
        params["before"] = last_page[-1]["created_utc"]

    results = requests.get(submissions_url, params)

    if not results.ok:
        # something wrong happened
        raise Exception("Server returned status code {}".format(results.status_code))

    return results.json()["data"]


def crawl_subreddit(subreddit: str, max_submissions: int = 2000, *, pause: int = 2):
    """
  Crawl submissions from a subreddit.

  :param subreddit: The subreddit to crawl.
  :param max_submissions: The maximum number of submissions to download.
  :param pause: Number of seconds to pause between each page fetch.

  :return: A list of submissions.
  """
    submissions:List[Dict] = []
    last_page = None

    while last_page != [] and len(submissions) < max_submissions:
        if submissions:
            # let's give the server a break between two page crawls
            time.sleep(pause)

        remaining = max_submissions - len(submissions)
        next_page_size = min(500, remaining)
        last_page = crawl_page(subreddit, last_page, page_size=next_page_size)
        submissions += last_page

    return submissions[:max_submissions]
