from nltk.corpus import wordnet as wn
import logging
import time
import requests

from projectxc6t73.config import config


def new_york_times_request(date, page, key):
    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    params = {
        'api-key': key,
        'begin_date': date,
        'end_date': date,
        'fl': 'keywords',
        'page': page
    }
    logging.info("Calling NYT API w/ params: %s" % str(params))
    return requests.get(url, params=params)


def daily_keywords(date, index):
    """
        Get 5 keywords for a particular day based on articles from New York Times
    Args:
        date: Day we should find keywords for (format:YYYYMMDD)
        index: Amount of calls done to the API
    Returns:
        list of keywords (ideally 5)
    """
    keywords = []
    calls = 0
    while len(keywords) < 5:
        time.sleep(1)

        data = {}
        try:
            data = new_york_times_request(date, str(calls), api_key(index)).json()
            logging.debug(str(data))
        except ValueError:
            logging.error('NYT API: JSONDecodeError')
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


def api_key(calls):
    if calls % 2 == 0:
        api_key = config['api_keys'][0]
    else:
        api_key = config['api_keys'][1]
    return api_key
