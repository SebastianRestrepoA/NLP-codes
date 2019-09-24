from text_processing_fns import *
import pandas as pd

path = './KnowledgeBase.xlsx'
vKnowledgeBase = pd.read_excel(path)

example_1 = count_words(vKnowledgeBase['Utterance'])
example_2 = remove_stopwords(vKnowledgeBase['Utterance'])
example_3 = remove_characters(vKnowledgeBase['Utterance'])
example_4 = lowercase_transform(vKnowledgeBase['Utterance'])
example_5 = lemmatization_transform(vKnowledgeBase['Utterance'])


vPathTaggedFile = 'C:/Users/Administrator/Documents/CHATBOT SOFY/Hipotecario/programas de gobierno/Refinamiento 1/' \
                    'Evaluacion inicial/Tagger_evaluacion_inicial.xlsx'

example_6 = fn_calculate_total_utterances_tagged(vPathTaggedFile)

vPathKnowledgeBase = 'C:/Users/Administrator/Documents/CHATBOT SOFY/Hipotecario/programas de gobierno/Refinamiento 1/' \
                    'Curacion/Programas de Gobierno.xlsx'

vPathSuccesFailFile = 'C:/Users/Administrator/Documents/CHATBOT SOFY/Hipotecario/programas de gobierno/Refinamiento 1/' \
                      'metrics/Iter_4/Kfolds/success_fail_confidence_programas_gobierno.xlsx'


example_7 = fn_calculate_total_utterances_per_intent(vPathKnowledgeBase, plot=True)

fn_utterances_similarity_between_intents(vPathKnowledgeBase, 0.8, 'similarity_analysis_programas')

example_8 = fn_calculate_word_frequency_per_intents(vPathKnowledgeBase, generate_excel=True)

fn_word_frequency_analysis_fail_utterances(vPathKnowledgeBase, vPathSuccesFailFile)

vPathNoAddUtterances = 'C:/Users/Administrator/Documents/CHATBOT SOFY/Hipotecario/programas de gobierno/Refinamiento 1/' \
                       'Curacion/utterances_no_agregadas.xlsx'

