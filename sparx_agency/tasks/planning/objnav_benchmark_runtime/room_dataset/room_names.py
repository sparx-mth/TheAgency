"""Room names as a model writes them, folded to one spelling so answers can be voted on."""
import collections
import re


def normalise_name(text):
    words = re.sub(r"[^a-z ]", " ", str(text or "").lower()).split()
    return " ".join(words[1:] if words[:1] in (["a"], ["an"], ["the"]) else words)


def majority_name(answers):
    """The most common name among the non-empty answers and its share of them; ties go to the alphabetically first name."""
    names = [n for n in map(normalise_name, answers) if n]
    if not names:
        return "", 0.0
    counts = collections.Counter(names)
    name = min(counts, key=lambda n: (-counts[n], n))
    return name, counts[name] / len(names)
