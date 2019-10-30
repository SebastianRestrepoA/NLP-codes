import os
import warnings
import numpy as np
from text_processing_fns import *
warnings.filterwarnings("ignore")
from classifiers import *
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

" LOAD DATA BASE"
vPathKnowledgeBase = './KnowledgeBase.xlsx'


#
KnowledgeBase = pd.read_excel(vPathKnowledgeBase)

# label encode the target variable to transform non-numerical labels
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(KnowledgeBase["Intent"])  # numerical labels
intent_names = encoder.classes_
nfolds = 10

" NATURAL LANGUAJE PROCESSING"

# transform our text information in lowercase
# KnowledgeBase["Utterance"] = lowercase_transform(KnowledgeBase["Utterance"])
#
# # Removing punctuation characters such as: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
# KnowledgeBase["Utterance"] = remove_characters(KnowledgeBase["Utterance"])
#
# # Removing stop words from text
# KnowledgeBase["Utterance"] = remove_stopwords(KnowledgeBase["Utterance"])

" WORD2VECT TRANSFORM AND FEATURE EXTRACTION"

features_matrix = feature_engineering(KnowledgeBase["Utterance"], tf_idf=True)


" MODEL DEVELOPMENT"

x = features_matrix['TF-IDF']['matrix']
best_svm_parameters = fn_search_best_svm_classifier(x, y, nfolds, 'TF-IDF', display_results=True)

best_kernel = best_svm_parameters['kernel'][0]
best_c = best_svm_parameters['C'][0]
best_gamma = best_svm_parameters['gamma'][0]


kf = StratifiedKFold(n_splits=10)
y_real_total = []
y_pred_total = []
fold = 1
fail_utterances = []

for train, val in kf.split(x, y):

    X_train, X_val, y_train, y_val = x[train], x[val], y[train], y[val]
    utterances_val, intent_names_val = KnowledgeBase["Utterance"][val], KnowledgeBase["Intent"][val]

    classifier = SVC(kernel=best_kernel, gamma=best_gamma, C=best_c).fit(X_train, y_train)

    # Model training
    classifier.fit(X_train, y_train)

    # Model prediction
    y_pred = classifier.predict(X_val)
    y_pred_total.append(y_pred)
    y_real_total.append(y_val)

    # to save the utterances that were not recognize correctly

    errors_idx = y_pred != y_val
    fail_utterances.append(pd.concat((utterances_val[errors_idx].reset_index(drop=True),
                                      intent_names_val[errors_idx].reset_index(drop=True),
                                      pd.DataFrame(encoder.inverse_transform(y_pred[errors_idx]),
                                                   columns=['Predicted intent'])),
                                     axis=1))

# The predictions obtained from cross validation procedure are concatenated and the performance measures are estimated.

y_real_total = np.concatenate(y_real_total)
y_pred_total = np.concatenate(y_pred_total)
metrics = classification_report(y_pred_total, y_real_total, target_names=intent_names, output_dict=True)
metrics.pop('micro avg', None)
metrics.pop('weighted avg', None)
fail_utterances = pd.concat(fail_utterances, ignore_index=True)

# save results in excel file
writer_metrics = pd.ExcelWriter('metrics_svm_model.xlsx', engine='xlsxwriter')
writer_fails = pd.ExcelWriter('fail_utterances_svm_model.xlsx', engine='xlsxwriter')

metrics = pd.DataFrame.from_dict(metrics, orient='index')
metrics.to_excel(writer_metrics)
fail_utterances.to_excel(writer_fails)
writer_fails.save()
writer_metrics.save()

