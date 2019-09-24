import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn import preprocessing
import nltk
import numpy as np
from keras import layers, models, optimizers
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import rcParams
from utils import *
import unicodedata
rcParams.update({'figure.autolayout': True})


def count_words(vKnowledgeBase):

    """ This function computes the number of words in each utterance belonging to the knowledge base.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with the number of words used in each utterance.


    """
    return vKnowledgeBase.apply(lambda x: len(str(x).split()))


def remove_stopwords(vKnowledgeBase):

    """ This function eliminates the stop words from each utterance belonging to the knowledge base.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with the knowledge base without stop words.


    """
    stop = stopwords.words('spanish')
    return vKnowledgeBase.apply(lambda x: " ".join(x for x in x.split() if x not in stop))


def remove_tilde(vUtterance):

    """ This function removes the accent mark (or "tilde") from a utterance.

    :param vUtterance: string variable with a utterance.

    :return: string variable with the utterance without "tildes".


    """

    return ''.join((c for c in unicodedata.normalize('NFD', vUtterance) if unicodedata.category(c) != 'Mn'))


def remove_characters(vKnowledgeBase):

    """ This function removes the irrelevant puntuation characters (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
    from each utterance belonging to the knowledge base.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with the knowledge base without irrelevant characters.


    """
    return vKnowledgeBase.str.replace('[^\w\s]', '')


def lowercase_transform(vKnowledgeBase):

    """ This function transforms the utterances belonging to the knowledge base to lowercase.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with the knowledge base transformed to lowercase.


    """

    return vKnowledgeBase.apply(lambda x: " ".join(x.strip().lower() for x in x.split()))


def text_lemmatization(utterance):

    """ This function apply lemmatization over all the words of a utterance.

    :param utterance: string variable.

    :return: string variable lemmatized.

    """

    words = word_tokenize(utterance)
    lemmatizer = WordNetLemmatizer()
    words_lemma = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words_lemma)


def lemmatization_transform(vKnowledgeBase):

    """ This function apply lemmatization in every utterance belonging to the knowledge base.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with lemmatized knowledge base.


    """

    return vKnowledgeBase.apply(lambda x: text_lemmatization(x))


def stemming_transform(vKnowledgeBase):

    """ This function apply stemming function in every utterance belonging to the knowledge base.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with stemmed knowledge base.

    """

    return vKnowledgeBase.apply(lambda x: text_stemming(x))


def text_stemming(utterance):

    """ This function apply stemming over all the words of a utterance.

    :param utterance: string variable.

    :return: stemmed string variable.

    """

    stemmer = SnowballStemmer("spanish")
    words = word_tokenize(utterance)
    words_stem = [stemmer.stem(word) for word in words]

    return ' '.join(words_stem)


def words_frequency(text, intent_name, plot=False):

    """ This function computes word frequency of a text.

    :param text: string variable with all concatenated utterances of a specific intent of the knowledge base.
           intent_name: string variable with the intent name in analysis.
           plot: this option allows to graph the frequency distribution of the words. Default, False

    :return: pandas dataframe variable with the frequency of every word in the knowledge base.

    """

    # break the string into list of words
    text_list = text.split()

    # gives set of unique words
    unique_words = pd.DataFrame(sorted(set(text_list)), columns=['word_names'])
    unique_words[intent_name] = unique_words['word_names'].apply(lambda x: text_list.count(x))
    unique_words = unique_words.sort_values(by=[intent_name], ascending=False).set_index('word_names')
    # unique_words.rename(columns={'Counts': intent_name})
    print(unique_words.head())

    if plot:
        unique_words.plot(kind='bar')
        plt.title(intent_name)
        plt.show()

    return unique_words


def lexical_diversity(text):

    """ This function  calculates lexical richness of a text.

    :param text: string variable.

    :return: float variable with the percentage of lexical diversity of the text in analysis.

    """
    text = text.split()
    return (len(set(text)) / len(text))*100


def extract_bigrams(utterance):

    """ This function produces bigrams list given a utterance.

    :param utterance: string variable.

    :return: list variable with the bigrams obtained from the input utterance.

    """
    utterance = utterance.split()
    return list(nltk.bigrams(utterance))


def extract_trigrams(utterance):

    """ This function produces trigrams list given a utterance.

    :param utterance: string variable.

    :return: list variable with the trigrams obtained from the input utterance.
    """

    utterance = utterance.split()
    return list(nltk.trigrams(utterance))


def intent_bigrams_frequency(vKnowledgeBase, intent_name):

    """ This function computes the frequency of bigrams in the knowledge base of specific intent.


    :param vKnowledgeBase: pandas series with the utterances of knowledge base.
           intent_name: string variable with the name of the intent in analysis.

    :return: pandas dataframe with the frequency of the bigrams in the knowledge base.


    """
    b = vKnowledgeBase.apply(lambda x: extract_bigrams(x))
    b = b.reset_index(drop=True)
    a = b[0]
    for i in range(1, len(b)):
        a = a + b[i]

    fdist = nltk.FreqDist(a)
    df = pd.DataFrame(fdist.items(), columns=['Bigrams', intent_name])

    return df.sort_values(by=[intent_name], ascending=False).set_index('Bigrams')


def intent_trigrams_frequency(vKnowledgeBase, intent_name):

    """ This function computes the frequency of trigrams in the knowledge base of specific intent.


    :param vKnowledgeBase: pandas series with the utterances of knowledge base.
           intent_name: string variable with the name of the intent in analysis.

    :return: pandas dataframe with the frequency of the trigrams in the knowledge base.


    """

    tri = vKnowledgeBase.apply(lambda x: extract_trigrams(x))
    tri = tri.reset_index(drop=True)
    a = tri[0]
    for i in range(1, len(tri)):
        a = a + tri[i]

    fdist = nltk.FreqDist(a)
    df = pd.DataFrame(fdist.items(), columns=['Trigrams', intent_name])
    return df.sort_values(by=[intent_name], ascending=False).set_index('Trigrams')


def fn_calculate_word_frequency_per_intents(path, delete_stop_word=False, stemming=False, lemmatization=False,
                                            delete_characters=False, generate_excel=False):

    """ This function computes the frequency of words, bigrams and trigrams in every intent of the knowledge base.


    :param path: string variable with local path where is saved the knowledge base.
           delete_stop_word: this option removes stop words of the knowledge base. Default False
           stemmming: this option applies stemming function over the knowledge base. Default False
           lemmatization: this option applies lemmatization function over the knowledge base. Default False
           delete_characters: this option deletes irrelevant puntuation characters over the knowledge base.
           Default False
           generate_excel: this option generates a excel file in the knowledge base path with the words, bigrams and
           trigrams frequency of the knowledge base. Default False

    :return: pandas dataframe with the words frequency of the knowledge base.


    """

    df = pd.read_excel(path)
    df["Intent"] = [remove_tilde(intent_name) for intent_name in df["Intent"]]
    unique_intent = list(set(df["Intent"]))

    # df["Utterance"] = df["Utterance"].apply(lambda row: row.strip().lower())

    # transform our text information in lowercase
    df["Utterance"] = lowercase_transform(df["Utterance"])

    if delete_characters:
        # Removing punctuation characters such as: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        df["Utterance_free_characters"] = remove_characters(df["Utterance_lower"])

    if delete_stop_word:
        # Removing stop words from text
        df["Utterance"] = remove_stopwords(df["Utterance"])

    if stemming:
        df["Utterance"] = stemming_transform(df["Utterance"])

    if lemmatization:
        df["Utterance"] = lemmatization_transform(df["Utterance"])

    lex_diversity = []

    df_old = []
    df_bigrams = []
    df_trigrams = []

    for idx, i in enumerate(unique_intent):

        if idx == 0:
            df_bigrams = intent_bigrams_frequency(df["Utterance"][df["Intent"] == i], i)
            df_trigrams = intent_trigrams_frequency(df["Utterance"][df["Intent"] == i], i)
            df_old = words_frequency(df['Utterance'][df["Intent"] == i].str.cat(sep=' '), i, plot=False)
        else:
            df_bigrams_new = intent_bigrams_frequency(df["Utterance"][df["Intent"] == i], i)
            df_trigrams_new = intent_trigrams_frequency(df["Utterance"][df["Intent"] == i], i)
            df_new = words_frequency(df['Utterance'][df["Intent"] == i].str.cat(sep=' '), i, plot=False)

            df_old = pd.concat((df_old, df_new), axis=1)
            df_bigrams = pd.concat((df_bigrams, df_bigrams_new), axis=1)
            df_trigrams = pd.concat((df_trigrams, df_trigrams_new), axis=1)

        lex_diversity.append(lexical_diversity(df['Utterance'][df["Intent"] == i].str.cat(sep=' ')))

    df_final = df_old.fillna(0).sort_values(by=unique_intent, ascending=False)
    bigrams_final = df_bigrams.fillna(0).sort_values(by=unique_intent, ascending=False)
    trigrams_final = df_trigrams.fillna(0).sort_values(by=unique_intent, ascending=False)
    df_lexical = pd.DataFrame(data=lex_diversity, index=df_final.columns, columns=['% distinct words'])
    df_lexical = df_lexical.sort_values(by=['% distinct words'], ascending=False)
    # df_final.style.apply(color_max_row, axis=1, subset=unique_intent)

    if generate_excel:
        excel_name = input('Indique el nombre del dominio de conocimiento del excel a generar: ')
        writer = pd.ExcelWriter(path[0:path.rfind('/') + 1] + 'Word frequency of ' + excel_name + '.xlsx', engine='xlsxwriter')
        df_final.to_excel(writer, sheet_name='word frequency')
        df_lexical.to_excel(writer, sheet_name='lexical diversity')
        bigrams_final.to_excel(writer, sheet_name='bigrams frequency')
        trigrams_final.to_excel(writer, sheet_name='trigrams frequency')
        writer.save()

    return df_final


def fn_word_frequency_analysis_fail_utterances(vPathKnowledgeBase, vPathSuccesFailFile):

    """ This function analyses the utterances that were not recognized by the machine learning model.
     It compares how many times is each word of the fail utterances in both the real and predict intent.

      :param vPathKnowledgeBase: string variable with local path where is saved the knowledge base (.xlsx file).
             vPathSuccesFailFile: string variable with local path where is saved success fail confidence file.


      :return: excel file with word frequency analysis over the utterances that were not recognized by
               machine learning model.

      """

    word_frequency = fn_calculate_word_frequency_per_intents(vPathKnowledgeBase)

    df_fail_groups = pd.read_excel(vPathSuccesFailFile, sheet_name='fail_groups').sort_values(by=['score'],
                                                                                              ascending=False)

    intents = list(set(df_fail_groups['real_intent']))
    writer = pd.ExcelWriter(vPathKnowledgeBase[0:vPathKnowledgeBase.rfind('/') + 1] +
                            'word_frequency_fail_utterances.xlsx', engine='xlsxwriter')

    for intent in intents:

        intent_fail = df_fail_groups[df_fail_groups['real_intent'] == intent].reset_index()

        all_tables = []
        for i in range(0, len(intent_fail)):

            all_tables.append(fn_get_word_frequency(word_frequency, intent_fail['utterance'][i], intent_fail['real_intent'][i],
                                                 intent_fail['pred_intent'][i]))

        multiple_dfs_to_excel(all_tables, writer, intent[:17], 5)

    writer.save()


def fn_calculate_total_utterances_per_intent(vPathKnowledgeBase, plot=False):

    """ This function computes the total of utterances per intent (or class) of the knowledge base.


    :param vPathKnowledgeBase: string variable with local path where is saved the knowledge base (.xlsx file)
           plot: boolean variable to make a horizontal bar plot with the total of utterances per intent. Default False.

    :return: pandas dataframe with the total of utterances per intent of the knowledge base.

    """

    df = pd.read_excel(vPathKnowledgeBase)

    intent_names = df['Intent'].unique()
    result = []

    for intent in intent_names:

        df_intent = df[df['Intent'] == intent]
        df_ong = [len(df_intent[df_intent['ongoing'] == ong]) for ong in df['ongoing'].unique()]
        result.append(df_ong)

    result = pd.DataFrame(result, index=intent_names, columns=['ongoing ' + str(i) for i in df['ongoing'].unique()])
    result['Total'] = result.sum(axis=1)

    result = result.sort_values(by='Total', ascending=False)

    if plot:

        ax = result.drop('Total', axis=1).plot(kind='barh', stacked=True)
        ax.set_xlabel("Number of Utterances")
        ax.set_title('Total of Utterances in the Knowledge Base')
        ax.set_xticks(range(0, result.max().max()+5, 5))

        for tick in ax.get_xticks():
            ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

        plt.show()

    return result


def fn_calculate_total_utterances_tagged(vPathTaggedFile, excel=False):

    """ This function computes the total of tagged utterances per intents of each domain.


    :param vPathTaggedFile: string variable with local path where is saved the tagged file (.xlsx file)

    :return: dictionary with the total of utterances per domain.

    """

    vTaggedFile = pd.read_excel(vPathTaggedFile)

    vResult = {}

    for domain_name in vTaggedFile['Dominio'].unique():

        vDomainData = vTaggedFile[vTaggedFile['Dominio'] == domain_name]

        vIntentData = [len(vDomainData[vDomainData['Intent'] == intent]) for intent in vDomainData['Intent'].unique()]

        vResult[domain_name] = pd.DataFrame(vIntentData, index=vDomainData['Intent'].unique(),
                                            columns=['Total utterances tagged']).\
            sort_values(by='Total utterances tagged', ascending=False)

    if excel is True:

        writer = pd.ExcelWriter(vPathTaggedFile[0:vPathTaggedFile.rfind('/') + 1] + 'total_utterances_tagged.xlsx',
                                engine='xlsxwriter')
        for domain, total_utterances_tagged in vResult.items():
            total_utterances_tagged.to_excel(writer, sheet_name=domain)

        writer.save()

    return vResult


def fn_get_word_frequency(df_wordfreq, utterance, real_intent, pred_intent):

    words = utterance.split(' ')

    return df_wordfreq.loc[words, [real_intent, pred_intent]]


def fn_get_jaccard_sim(vStr1, vStr2):

    """ This function computes the jaccard similarity coefficient between two utterances (or strings).

    :param vStr1: string variable.
           vStr2: string variable.
    :return float variable with the jaccard similarity coefficient.

    """
    a = set(vStr1.split())
    b = set(vStr2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def fn_get_vectors(strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def fn_get_cosine_sim(vUtterancesKnowledgeBase):

    """ This function computes the cosine similarity between the utterances of the knowledge base.

    :param vUtterancesKnowledgeBase: list variable with the utterances of the knowledge base (.xlsx file)

    :return ndarray matrix with the cosine similarity estimated from the knowledge base.

    """

    vectors = [t for t in fn_get_vectors(vUtterancesKnowledgeBase)]
    return cosine_similarity(vectors)


def fn_utterances_similarity_between_intents(vPathKnowledgeBase, vThreshold, output_file_name):

    """ This function computes the jaccard and cosine similarity index between the utterances of the knowledge base and
     it generates a excel file with the utterances that achieved an jaccard index above setted threshold.


    :param vPathKnowledgeBase: string variable with local path where is saved the knowledge base (.xlsx file)
           vThreshold: float variable used to filter the utterances with similarity below of this number.

    """

    writer = pd.ExcelWriter(vPathKnowledgeBase[0:vPathKnowledgeBase.rfind('/') + 1] +
                            output_file_name + '.xlsx', engine='xlsxwriter')

    df = pd.read_excel(vPathKnowledgeBase)

    df['Utterance'] = lowercase_transform(df['Utterance'])
    df = df.reset_index()
    # df = df[df.columns[:-2]]
    jaccard_matrix = np.zeros((len(df['Utterance']), len(df['Utterance'])))

    for idx, utterance in enumerate(list(df['Utterance'])):
        jaccard_matrix[:, idx] = df['Utterance'].apply(lambda x: fn_get_jaccard_sim(utterance, x))

    cosine_matrix = fn_get_cosine_sim(list(df['Utterance']))

    df_final = []

    for idx in range(0, df.shape[0]):

        idx_utterances = np.where((jaccard_matrix[idx+1:, idx] > vThreshold) == 1)
        idx_utterances = idx_utterances[0]+idx+1
        utterances_selected = df.iloc[idx_utterances].reset_index(drop=True)
        utterance_to_compare = pd.DataFrame([df['Utterance'][idx]] * utterances_selected.shape[0],
                                            columns=['To compare'])
        intent_to_compare = pd.DataFrame([df['Intent'][idx]] * utterances_selected.shape[0],
                                         columns=['Intent to compare'])
        jaccard_data = pd.DataFrame(jaccard_matrix[idx_utterances, idx], columns=['Jaccard'])
        cosine_data = pd.DataFrame(cosine_matrix[idx_utterances, idx], columns=['Cosine'])

        if utterances_selected is not None:
            df_contenated = pd.concat((intent_to_compare, utterance_to_compare, utterances_selected, jaccard_data,
                                       cosine_data), axis=1).sort_values(by='Jaccard', ascending=False)

            df_final.append(df_contenated[['Intent to compare', 'To compare', 'Utterance', 'Intent', 'Jaccard', 'Cosine',
                               'index', 'ongoing']])

    df_final = pd.concat(df_final)
    df_final.to_excel(writer)

    writer.save()
    print('Similarity between intents have finished')


def feature_engineering(vUtterances, vectorizer=False, tf_idf=False, ngram=False):

    """ This function coverts utterances into feature matrices such as word2vect, TF-IDF and n-gram for training
    of ML models.

    :param vUtterances: pandas series with the utterances of knowledge base.

    :return: dictionary with 3 features matrices corresponding to the count vector, TF-IDF and n-grams transforms.

    """

    features = {}
    vUtterances.index = range(0, len(vUtterances))

    if vectorizer is True:

        # transform the training data using count vectorizer object
        # create a count vectorizer object
        count_vect = CountVectorizer(analyzer='word')
        count_vect.fit(vUtterances) # Create a vocabulary from all utterances
        x_count = count_vect.transform(vUtterances)  # Count how many times is each word from each utterance in the
        # vocabulary.
        features['count_vectorizer'] = {'object': count_vect, 'matrix': x_count}
        # pd.DataFrame(x_count.toarray(), columns=count_vect.get_feature_names())

    if tf_idf is True:

        # word level tf-idf

        " TF-IDF score represents the relative importance of a term in the document and the entire corpus. "
        tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000)  # token_pattern=r'\w{1,}'
        tfidf_vect.fit(vUtterances)
        x_tfidf = tfidf_vect.transform(vUtterances)
        features['TF-IDF'] = {'object': tfidf_vect, 'matrix': x_tfidf}

    if ngram is True:

        # ngram level tf-idf
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=5000) # token_pattern=r'\w{1,}'
        tfidf_vect_ngram.fit(vUtterances)
        x_tfidf_ngram = tfidf_vect_ngram.transform(vUtterances)
        features['ngram'] = {'object': tfidf_vect_ngram, 'matrix': x_tfidf_ngram}

    return features


def fn_utterances_similarity_two_docs(path_doc1, path_doc2, output_file_name):

    writer = pd.ExcelWriter(output_file_name + '.xlsx')
    utterances_knowledge_base = pd.read_excel(path_doc1)
    utterances_knowledge_base['Utterance'] = lowercase_transform(utterances_knowledge_base['Utterance'])
    utterances_knowledge_base = utterances_knowledge_base.reset_index(drop=True)
    utterances_knowledge_base = utterances_knowledge_base[utterances_knowledge_base.columns[:-2]]

    utterances_doc_to_depure = pd.read_excel(path_doc2)
    utterances_doc_to_depure['Utterance'] = lowercase_transform(utterances_doc_to_depure['Utterance'])
    utterances_doc_to_depure = utterances_doc_to_depure.reset_index(drop=True)
    # utterances_doc_to_depure = utterances_doc_to_depure[utterances_doc_to_depure.columns[:-2]]

    id = utterances_doc_to_depure['Utterance'].isin(utterances_knowledge_base['Utterance'])
    new_doc = utterances_doc_to_depure[id == False]
    # new_doc = utterances_doc_to_depure[id == True]

    writer = pd.ExcelWriter('utterances_No_agregadas.xlsx', engine='xlsxwriter')
    new_doc.to_excel(writer, index=False)
    writer.save()


def create_nn_model_architecture(input_size):
    # create input layer
    input_layer = layers.Input((input_size,), sparse=True)

    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)

    # create output layer
    output_layer = layers.Dense(11, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier



