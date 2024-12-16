import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from categories import COLORS, DAYS, MONTHS, GREETINGS, VEHICLES, CATEGORIES
from nltk.tokenize import sent_tokenize, word_tokenize 
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')

def preprocess_to_isl(input_text):
    # """
    # Converts English text into ISL (Sinhala Sign Language) compatible structure.
    # """
    # stop_words = set(stopwords.words("english"))
    # lemmatizer = WordNetLemmatizer()

    # words = nltk.word_tokenize(input_text)
    # pos_tags = nltk.pos_tag(words)

    # isltree = Tree('ROOT', [])

    # for word, tag in pos_tags:
    #     word_lower = word.lower()

    #     if tag in CATEGORIES["nouns"] or word_lower in VEHICLES: 
    #         isltree.append(word)
    #     elif tag in CATEGORIES["verbs"]:  # Verbs
    #         isltree.append(word)
    #     elif tag in CATEGORIES["adjectives"] or word_lower in COLORS:  # Adjectives or colors
    #         isltree.append(word)
    #     elif tag in CATEGORIES["adverbs"]:  # Adverbs
    #         isltree.append(word)
    #     elif word_lower in DAYS or word_lower in MONTHS:  # Days or months
    #         isltree.append(word)
    #     elif word_lower in GREETINGS:  # Greetings
    #         isltree.append(word)
    #     elif tag in CATEGORIES["prepositions"]:  # Prepositions
    #         isltree.append(word)
    #     elif tag in CATEGORIES["conjunctions"]:  # Conjunctions
    #         isltree.append(word)
    #     elif tag in CATEGORIES["numbers"]:  # Numbers
    #         isltree.append(word)
    #     elif tag in CATEGORIES["interjections"]:  # Interjections
    #         isltree.append(word)

    # # Filter and lemmatize words
    # filtered_words = [
    #     lemmatizer.lemmatize(word) for word in isltree.leaves() if word.lower() not in stop_words
    # ]
    # isl_sentence = " ".join(filtered_words)
    tokens = word_tokenize(input_text)
    isl_sentence = [token.capitalize() for token in tokens]
    return " ".join(isl_sentence)

