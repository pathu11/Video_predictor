# categories.py

COLORS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown","gold","gray"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday","tomorrow","week","yesterday","today","seconds","night","morning","hour","good night","good morning","good evening","evening","day","day after tomorrow"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
GREETINGS = ["hello", "hi", "hey", "greetings", "welcome","alright","aubowan","how are you","thank you"]
VEHICLES = ["car", "bike", "bus", "train", "plane", "ship", "truck", "bicycle", "van", "scooter"]

# POS tags for different categories
CATEGORIES = {
    "nouns": ["NN", "NNS", "NNP", "NNPS"],
    "verbs": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    "adjectives": ["JJ", "JJR", "JJS"],
    "adverbs": ["RB", "RBR", "RBS"],
    "prepositions": ["IN", "TO"],
    "conjunctions": ["CC"],
    "numbers": ["CD"],
    "interjections": ["UH"]
}
