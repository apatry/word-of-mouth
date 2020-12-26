"""
Minimalistic reddit crawler.

This crawler leverage the
[pushshift api](https://github.com/pushshift/api).
"""

import time
from typing import Dict, List

import requests

# Pushshift end point serving submissions
submissions_url = "https://api.pushshift.io/reddit/search/submission"


def _crawl_page(subreddit: str, last_page=None, *, page_size: int = 500):
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


def crawl_subreddit(
    subreddit: str, max_posts: int = 2000, *, pause: int = 2,
):
    """
    Crawl posts from a subreddit.

    :param subreddit: The subreddit to crawl.
    :param max_posts: The maximum number of posts to download.
    :param pause: Number of seconds to pause between each page fetch.

    :return: A list of posts.
    """
    posts: List[Dict] = []
    last_page = None

    while last_page != [] and len(posts) < max_posts:
        if posts:
            # let's give the server a break between two page crawls
            time.sleep(pause)

        remaining = max_posts - len(posts)
        next_page_size = min(500, remaining)
        last_page = _crawl_page(subreddit, last_page, page_size=next_page_size)
        posts += last_page

    return posts[:max_posts]
