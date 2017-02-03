from nltk.corpus import wordnet as wn
from collections import Counter
import logging
import os
import re


def words(text): return re.findall(r'\w+', text.lower())


WORDS = Counter(words(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "million_words.txt")).read()))


def semantic_similarity(node, other):
    """Semantic similarity(i.e. edge counting) of a WordNet word to another word"""
    node_synset = synset(node)
    other_synset = synset(other)
    if node_synset and other_synset:
        return node_synset.path_similarity(other_synset)
    else:
        return 0.0


def synset(node):
    """WordNet synset"""
    synsets = wn.synsets(node)
    if synsets:
        return synsets[0]
    else:
        return default_synset(node)


def default_synset(node):
    """Correctly/similarly spelled word synset"""
    synsets = wn.synsets(correction(node))
    if synsets:
        return synsets[0]
    else:
        return None


def P(word, N=sum(WORDS.values())):
    """Probability of `word`."""
    return WORDS[word] / N


def correction(word):
    """Most probable spelling correction for word."""
    corr = max(candidates(word), key=P)
    logging.info("Spelling correction: %s -> %s" % (word, corr))
    return corr


def candidates(word):
    """Generate possible spelling corrections for word."""
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    """The subset of `words` that appear in the dictionary of WORDS."""
    return set(w for w in words if w in WORDS)


def edits1(word):
    """All edits that are one edit away from `word`."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """All edits that are two edits away from `word`."""
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
