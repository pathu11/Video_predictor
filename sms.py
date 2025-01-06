
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')


def get_synonyms(word):
    """
    Retrieve synonyms for a given word using WordNet.
    
    Args:
        word (str): The word to find synonyms for.
    
    Returns:
        list: A list of synonyms for the given word.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().capitalize()) 
    return list(synonyms)

print(get_synonyms("Smell"))

