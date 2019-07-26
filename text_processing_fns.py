import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn import preprocessing
import nltk
import  numpy as np
# from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def count_words(vKnowledgeBase):

    """ This function computes the number of words in each utterance belonging to the knowledge base.

    :param vKnowledgeBase: pandas series with the utterances of the knowledge base.

    :return: pandas series with the number of words used in each utterance.


    """
    return vKnowledgeBase.apply(lambda x: len(str(x).split()))


def remove_stopwords(text):

    stop = stopwords.words('spanish')
    return text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))


def remove_characters(text):

    return text.str.replace('[^\w\s]', '')


def lowercase_transform(text):

    return text.apply(lambda x: " ".join(x.strip().lower() for x in x.split()))


def text_lemmatization(text):

    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words_lemma = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words_lemma)


def lemmatization_transform(text):

    return text.apply(lambda x: text_lemmatization(x))


def stemming_transform(text):

    return text.apply(lambda x: text_stemming(x))


def text_stemming(text):

    stemmer = SnowballStemmer("spanish")
    words = word_tokenize(text)
    words_stem = [stemmer.stem(word) for word in words]

    return ' '.join(words_stem)


def words_frequency(text, intent_name, plot=False):

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
    text = text.split()
    return (len(set(text)) / len(text))*100


def extract_bigrams(utterance):
    utterance = utterance.split()
    return list(nltk.bigrams(utterance))


def extract_trigrams(utterance):
    utterance = utterance.split()
    return list(nltk.trigrams(utterance))


def intent_bigrams_frequency(utterances, intent_name):
    b = utterances.apply(lambda x: extract_bigrams(x))
    b = b.reset_index(drop=True)
    a = b[0]
    for i in range(1, len(b)):
        a = a + b[i]

    fdist = nltk.FreqDist(a)
    df = pd.DataFrame(fdist.items(), columns=['Bigrams', intent_name])
    df = df.sort_values(by=[intent_name], ascending=False).set_index('Bigrams')
    return df


def intent_trigrams_frequency(utterances, intent_name):
    tri = utterances.apply(lambda x: extract_trigrams(x))
    tri = tri.reset_index(drop=True)
    a = tri[0]
    for i in range(1, len(tri)):
        a = a + tri[i]

    fdist = nltk.FreqDist(a)
    df = pd.DataFrame(fdist.items(), columns=['Trigrams', intent_name])
    df = df.sort_values(by=[intent_name], ascending=False).set_index('Trigrams')
    return df


def feature_engineering(x):

    features = {}
    x.index = range(0, len(x))

    # transform the training data using count vectorizer object
    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word')
    count_vect.fit(x) # Create a vocabulary from all utterances
    x_count = count_vect.transform(x) # Count how many times is each word from each utterance in the vocabulary.
    features['count_vectorizer'] = {'object': count_vect, 'matrix': x_count}
    # pd.DataFrame(x_count.toarray(), columns=count_vect.get_feature_names())

    # word level tf-idf

    " TF-IDF score represents the relative importance of a term in the document and the entire corpus. "
    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000)  # token_pattern=r'\w{1,}'
    tfidf_vect.fit(x)
    x_tfidf = tfidf_vect.transform(x)
    features['TF-IDF'] = {'object': tfidf_vect, 'matrix': x_tfidf}

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=5000) # token_pattern=r'\w{1,}'
    tfidf_vect_ngram.fit(x)
    x_tfidf_ngram = tfidf_vect_ngram.transform(x)
    features['ngram'] = {'object': tfidf_vect_ngram, 'matrix': x_tfidf_ngram}

    return features


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

def color_max_row(row):
    return ['background-color: red' if x > 3 else 'background-color: yellow' for x in row]


def row_f1Score_color(row):
    # vResult = vResult.style.apply(ro_f1Score_color, axis=1)

    if (row["f1-score"] >= 0.85):
        return pd.Series('background-color: #41DF26', row.index)
    elif (row["f1-score"] < 0.7):
        return pd.Series('background-color: red', row.index)
    else:
        return pd.Series('background-color: yellow', row.index)


def fn_calculate_total_utterances_per_intent(path, plot=False):

    df = pd.read_excel(path)

    intent_names = df['Intent'].unique()
    result = []

    for intent in intent_names:

        df_intent = df[df['Intent'] == intent]
        df_ong = []

        for ong in df['ongoing'].unique():
            df_ong.append(len(df_intent[df_intent['ongoing'] == ong]))

        result.append(df_ong)
    result = pd.DataFrame(result, index=intent_names, columns=['Training', 'Reto 1', 'Reto 2', 'Reto 3'])
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


def fn_calculate_total_utterances_per_domains(path, plot=False):

    df = pd.read_excel(path)

    for domain in ['cesantias', 'office']:

        df_domain = df[df['Dominio']==domain]
        intent_names = df_domain['Intent'].unique()
        result = []

        for intent in intent_names:

            df_intent = df[df['Intent'] == intent]
            df_ong = []

            for ong in df['ongoing'].unique():
                df_ong.append(len(df_intent[df_intent['ongoing'] == ong]))

            result.append(df_ong)
        result = pd.DataFrame(result, index=intent_names, columns=['Reto 1', 'Reto 2', 'Reto 3'])
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


def fn_calculate_word_frequency_per_intents(path, delete_stop_word=False, stemming=False, lemmatization=False,
                                            delete_characters=False, generate_excel=False):
    df = pd.read_excel(path)
    intent = df["Intent"]
    unique_intent = list(set(intent))

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
        writer = pd.ExcelWriter('Word frequency of ' + excel_name + '.xlsx', engine='xlsxwriter')
        df_final.to_excel(writer, sheet_name='word frequency')
        df_lexical.to_excel(writer, sheet_name='lexical diversity')
        bigrams_final.to_excel(writer, sheet_name='bigrams frequency')
        trigrams_final.to_excel(writer, sheet_name='trigrams frequency')
        writer.save()

    return df_final


def get_word_frequency(df_wordfreq, utterance, real_intent, pred_intent):

    words = utterance.split(' ')

    return df_wordfreq.loc[words, [real_intent, pred_intent]]


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_vectors(strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def get_cosine_sim(strs):
    vectors = [t for t in get_vectors(strs)]
    return cosine_similarity(vectors)


def fn_utterances_similarity_per_intent(path, threshold):

    df = pd.read_excel(path)

    df['Utterance'] = lowercase_transform(df['Utterance'])
    intents = list(set(df['Intent']))

    writer = pd.ExcelWriter('similitud por intencion1.xlsx')

    for intent in intents:

        try:

            a = df[df['Intent'] == intent].reset_index()
            a = a[['Utterance', 'ongoing', 'index']]

            jaccard_matrix = np.zeros((len(a['Utterance']), len(a['Utterance'])))

            for idx, utterance in enumerate(list(a['Utterance'])):

                jaccard_matrix[:, idx] = a['Utterance'].apply(lambda x: get_jaccard_sim(utterance, x))

            cosine_matrix = get_cosine_sim(list(a['Utterance']))

            df_final = []

            for idx in range(0, a.shape[0]):

                idx_utterances = np.where((jaccard_matrix[:, idx] > threshold) == 1)
                idx_utterances = idx_utterances[0][idx+1:]
                utterances_selected = a.iloc[idx_utterances].reset_index(drop=True)
                utterance_to_compare = pd.DataFrame([a.iloc[idx, 0]]*utterances_selected.shape[0], columns=['To compare'])
                jaccard_data = pd.DataFrame(jaccard_matrix[idx_utterances, idx], columns=['Jaccard'])
                cosine_data = pd.DataFrame(cosine_matrix[idx_utterances, idx], columns=['Cosine'])

                if utterances_selected is not None:

                    df_contenated = pd.concat((utterance_to_compare, utterances_selected, jaccard_data, cosine_data), axis=1)\
                        .sort_values(by='Jaccard', ascending=False)
                    df_final.append(df_contenated[['To compare', 'Utterance', 'Jaccard', 'Cosine', 'index', 'ongoing']])

            df_final = pd.concat(df_final)
            df_final.to_excel(writer, sheet_name=intent[:10])
        except:

            print('¡¡¡ Danger !!!: a problem has ocurred with the intent: ' + intent)

    print('Similarity per intent have finished')

    writer.save()


def fn_utterances_similarity_between_intents(path, threshold, output_file_name):

    writer = pd.ExcelWriter(output_file_name + '.xlsx')

    df = pd.read_excel(path)

    df['Utterance'] = lowercase_transform(df['Utterance'])
    df = df.reset_index()
    # df = df[df.columns[:-2]]
    jaccard_matrix = np.zeros((len(df['Utterance']), len(df['Utterance'])))

    for idx, utterance in enumerate(list(df['Utterance'])):
        jaccard_matrix[:, idx] = df['Utterance'].apply(lambda x: get_jaccard_sim(utterance, x))

    cosine_matrix = get_cosine_sim(list(df['Utterance']))

    df_final = []

    for idx in range(0, df.shape[0]):

        idx_utterances = np.where((jaccard_matrix[idx+1:, idx] > threshold) == 1)
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
