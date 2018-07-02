import nltk
from nltk import word_tokenize, sent_tokenize, RegexpTokenizer
from stop_list import closed_class_stop_words
import re

def process_sentence(raw_sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    sentence_without_numbers = re.sub(r'\d+', '', raw_sentence)
    tokens = tokenizer.tokenize(sentence_without_numbers)

    # removing stopwords
    cleaned_sentence = []
    for w in tokens:
        if w not in closed_class_stop_words:
            cleaned_sentence.append(w)
    return cleaned_sentence

sentence = 'hey my name is mir ahmed. I go to nyu. I love nyc. nyc is the best city in the world and it is located in 42 street'
result = process_sentence(sentence)
#print(result)