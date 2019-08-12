
import pandas as pd
from utils import fn_sofy_response
import json
from sklearn.metrics import classification_report

with open('config.json', 'r') as f:
    authoring_key = json.load(f)

authoring_key = authoring_key['authoring_key']

vEndPoints = {'MainRouter': 'https://eastus2.api.cognitive.microsoft.com/luis/v2.0/apps/2fa5c5f7-0abf-4f51-8c28-a11df71'
                            'bbcb9',

              'Cesantias': 'https://eastus2.api.cognitive.microsoft.com/luis/v2.0/apps/b4628efb-8e92-468c-be64-af58b8'
                           'f389e4',

              'HipotecarioRouter': 'https://eastus2.api.cognitive.microsoft.com/luis/v2.0/apps/52f4defd-0101-4b57-a463-'
                                   '84652141b580',

              'ProgramasGobierno': 'https://eastus2.api.cognitive.microsoft.com/luis/v2.0/apps/f4abd5ee-616d-4705-9c68'
                                   '-9d80e2a36dbe',

              'Office': 'https://eastus2.api.cognitive.microsoft.com/luis/v2.0/apps/69815345-5eca-4387-b167-d05111ceb'
                        'c1c'}

# utt = 'en que puedo usar las cesantias'
#
# aa = fn_sofy_response(vEndPoints, autoring_key, utt)

# vKnowledgeBasePaths = {'main_router': './Architecture validation knowledge base/main_router_test.xlsx',
#                        'programas_gobierno': './Architecture validation knowledge base/programas_gobierno_test.xlsx',
#                        'compra_cartera': './Architecture validation knowledge base/compra_de_cartera_test.xlsx',
#                        'cesantias': './Architecture validation knowledge base/cesantias_test.xlsx',
#                        'office': './Architecture validation knowledge base/office_test.xlsx',
#
#                        }

vKnowledgeBasePaths = {'cesantias': 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/'
                                    'Refinamiento_4.1/Curacion/Cesantias/Cesantías_test.xlsx'}

writer_metrics = pd.ExcelWriter('metrics_cognitive_architecture.xlsx', engine='xlsxwriter')
writer_fails = pd.ExcelWriter('fail_utterances_cognitive_paths.xlsx', engine='xlsxwriter')

for key in vKnowledgeBasePaths:

    vKnowledgeBase = pd.read_excel(vKnowledgeBasePaths[key])
    y_pred = [fn_sofy_response(vEndPoints, autoring_key, utterance) for utterance in vKnowledgeBase['Utterance']]
    utterances_paths = pd.concat(y_pred, ignore_index=True, sort=False)
    utterances_paths = pd.concat((vKnowledgeBase['Utterance'], utterances_paths), axis=1)
    utterances_paths = pd.concat((utterances_paths, vKnowledgeBase['Intent']), axis=1)
    utterances_paths.rename(columns={'Intent': 'real_intent'}, inplace=True)

    fail_utt = utterances_paths[utterances_paths['final_output'] != utterances_paths['real_intent']]
    metrics = classification_report(utterances_paths['real_intent'], utterances_paths['final_output'], output_dict=True)
    metrics.pop('weighted avg', None)
    metrics.pop('micro avg', None)
    final_metrics = pd.DataFrame.from_dict(metrics, orient='index')
    final_metrics.to_excel(writer_metrics, sheet_name=key)
    fail_utt.to_excel(writer_fails,  sheet_name=key)

writer_fails.save()
writer_metrics.save()

