import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from transformers import pipeline
summarizer = pipeline("summarization", model="t5-small", framework="pt")


def summarize_text_advanced(text):
    if len(text.split()) <= 5:
        # If the text has 5 or fewer words, skip summarization
        return text
    summarized = summarizer(text, max_length=90, min_length=10, do_sample=False)
    return summarized[0]['summary_text']

def preprocess_to_isl(sentence):
    summarized_sentence = summarize_text_advanced(sentence)
    print("Summarized Text:", summarized_sentence)

    tokens = word_tokenize(summarized_sentence)
    print("Tokens:", tokens)

    # POS Tagging (Part of Speech tagging)
    tagged_tokens = pos_tag(tokens)
    print("POS Tagged Tokens:", tagged_tokens)

    # List of stop words (excluding pronouns)
    stop_words = set(stopwords.words('english'))
    allowed_pos = {'NN', 'NNS', 'NNPS', 'NNP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'PRP', 'PRP$', 'DT', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RB$'}

    # Filtering the tokens based on POS and excluding stopwords
    filtered_tokens_with_pos = [
        (word, pos) for word, pos in tagged_tokens 
        if (pos in allowed_pos and word.lower() not in stop_words) or pos in ['PRP', 'PRP$']
    ]
    print("Filtered Tokens (Kept relevant words):", filtered_tokens_with_pos)

    # Lemmatization (reducing verbs to their base form)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, pos='v') if pos.startswith('V') else word
        for word, pos in filtered_tokens_with_pos
    ]
    print("Lemmatized Tokens:", lemmatized_tokens)

    # Capitalizing the first letter of each token (optional based on your requirement)
    isl_sentence = [word.capitalize() for word in lemmatized_tokens]
    print("Concatenated string:", " ".join(isl_sentence))

    return " ".join(isl_sentence)
