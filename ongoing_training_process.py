import pandas as pd
from utils import *


''' -----------------------------  GENERATE AUTO AND MANUAL TAGGED FILES  --------------------------------- '''
vLogsPath = 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_4/' \
            'Logs_test2.xlsx'

vAppsID = {"Enrutador": "c771ca51-5870-4b6a-b9b2-d86e5a0aa950", "Cesantias": "39bdf96c-c04d-4ccc-bd2c-731243b3f8b3",
           "Office": "7837b46b-8232-4429-8ea3-24b1adbb7850"}

vKnowledgeBasePaths = {'cesantias': 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_3/'
                                    'Curacion/Cesantias/iter4/cesantias_test.xlsx',
                       'office': 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_3/'
                                 'Curacion/Office/office_test.xlsx',
                       'enrutador': 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_3/'
                                    'Curacion/Enrutador/enrutador_test.xlsx',
                       }

vNoAddPaths = {'cesantias': 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_3/'
                            'Curacion/Cesantias/iter4/cesantias_test_no_add.xlsx',
               'office': 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_3/'
                         'Curacion/Office/office_test_no_add.xlsx',
               'enrutador': 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_3/'
                            'Curacion/Enrutador/enrutador_test_no_add.xlsx'}

fn_generate_tagger_file(vLogsPath, vAppsID, vKnowledgeBasePaths, vNoAddPaths)

'''--------------------- MERGE MANUAL TAGGED  AND TAGGED HIDE FILE --------------------------------------'''

vManualTaggedPath = 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_4/Tagger0.xlsx'
vTaggedHidePath = 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_4/Tagger_Hide0.xlsx'
vManualTagged = pd.read_excel(vManualTaggedPath, sheet_name="Sheet1")
vTaggedHide = pd.read_excel(vTaggedHidePath)
vMergeManualTaggedDataFrame = fn_merge_tagged_formats(vManualTagged, vTaggedHide)

'''--------------------- CONCATENATE MANUAL AND AUTO TAGGED FILES --------------------------------------'''

vMergeAutoTaggedPath = 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_4/' \
                       'Merge_Auto_Tagger.xlsx'

vMergeAutoTaggedDataFrame = pd.read_excel(vMergeAutoTaggedPath)
vMergeTaggedDataFrame = pd.concat((vMergeManualTaggedDataFrame, vMergeAutoTaggedDataFrame))

'''-------------------- GENERATE EXCEL FILE TO COMPUTE THE INITIAL EVALUATION OF THE MAIN ROUTER --------'''
# vBagsMapping = {'office': 'aplicaciones', 'cesantias': 'administrar_novedades_de_nomina'}
# vOmitBags = ['s_n', 'descarte', 'naturalidad']

vBagsMapping = {'office': 'aplicaciones', 'cesantias': 'administrar_novedades_de_nomina',
                'descarte': 'None', 'naturalidad': 'None', 's_n': 'None'}
vThresholdFlag = True

vOmitBags = []
vRouterPredictions = fn_get_router_valPredictions(vMergeTaggedDataFrame, vOmitBags, vBagsMapping, vLogsPath,
                                                  saveFlag=True)

'''---------------------- GENERATE EXCEL FILE TO COMPUTE THE INITIAL EVALUATION OF THE CHILDS -------------'''
vChildTags = ['cesantias', 'office']
vChildsUpdateDict = fn_get_childs_valPredictions(vMergeTaggedDataFrame, vChildTags, vLogsPath, none=True, saveFlag=True)
