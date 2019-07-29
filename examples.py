from text_processing_fns import *
import pandas as pd

utterance = 'Esto es una prueba de funciones de procesamiento de lenguaje'

path = './KnowledgeBase.xlsx'
vKnowledgeBase = pd.read_excel(path)

a = count_words(vKnowledgeBase['Utterance'])
b = remove_stopwords(vKnowledgeBase['Utterance'])
c = remove_characters(vKnowledgeBase['Utterance'])
d = lowercase_transform(vKnowledgeBase['Utterance'])
e = lemmatization_transform(vKnowledgeBase['Utterance'])


fn_calculate_word_frequency_per_intents(path)

print()

