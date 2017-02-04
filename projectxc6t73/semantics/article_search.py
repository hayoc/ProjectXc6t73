from nltk.corpus import wordnet as wn
import logging
import time
import requests


def new_york_times_request(date, page):
    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    params = {
        'api-key': '146f93e3db7345cc885b14999e9833d8',
        'begin_date': date,
        'end_date': date,
        'fl': 'keywords',
        'page': page
    }
    logging.info("Calling NYT API w/ params: %s" % str(params))
    return requests.get(url, params=params)


def daily_keywords(date):
    """
        Get 5 keywords for a particular day based on articles from New York Times
    Args:
        date: Day we should find keywords for (format:YYYYMMDD)
    Returns:
        list of keywords (ideally 5)
    """
    keywords = []
    calls = 0
    while len(keywords) < 5:
        time.sleep(1)
        data = new_york_times_request(date, str(calls)).json()
        logging.debug(str(data))
        if 'response' not in data:
            logging.error('NYT API limit exceeded')
            return keywords
        calls += 1

        for result in data['response']['docs']:
            for keyword in result['keywords']:
                if keyword['name'] == 'subject':
                    value = keyword['value']
                    word = value.split()[0]
                    if wn.synsets(word):
                        keywords.append(word)
                    if len(keywords) >= 5:
                        return keywords
        # Stop if we've called API more than 10 times and still haven't found enough keywords
        if calls > 10:
            logging.warning('Unable to find 5 keywords on %s' % date)
            return keywords


