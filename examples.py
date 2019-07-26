from text_processing_fns import *
import pandas as pd

utterance = 'Esto es una prueba de funciones de procesamiento de lenguaje'

vKnowledgeBase = pd.read_excel('./KnowledgeBase.xlsx')

a = count_words(vKnowledgeBase['Utterance'])