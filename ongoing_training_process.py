import pandas as pd
from utils import *


''' -----------------------------  GENERATE AUTO AND MANUAL TAGGED FILES  --------------------------------- '''
vLogsPath = 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_5/logs_conversacionales.xlsx'

vAppsID = {"Enrutador": "a550e46b-6d59-404b-aae6-9bd4a8ec6288",
           "Cesantias": "5e76e01a-1eeb-493b-8e61-5aba2c4abca5",
           "Office": "cbc4de50-e642-4872-813c-8784bd704af8",
           "Programas_gobierno": "ca30efab-ff99-4a4d-b200-eedb7d4c6b7f"}


vKnowledgeBasePaths = {'cesantias': 'C:/Users/Administrator/Documents/CHATBOT SOFY/CHATBOT SOFY/'
                                    'Gestion humana y administrativa/Cesantias/Produccion/cesantias_produccion.xlsx',
                       'office': 'C:/Users/Administrator/Documents/CHATBOT SOFY/CHATBOT SOFY/Tecnologia/Outlook/'
                                 'Produccion/office_produccion.xlsx',
                       'enrutador': 'C:/Users/Administrator/Documents/CHATBOT SOFY/CHATBOT SOFY/Enrutador principal/'
                                    'Produccion/Enrutador_produccion.xlsx',
                       'programas_gobierno': 'C:/Users/Administrator/Documents/CHATBOT SOFY/CHATBOT SOFY/Producto/'
                                             'Programas_de_gobierno/Produccion/Programas de Gobierno_120919.xlsx'
                       }

vNoAddPaths = {'cesantias': 'C:/Users/Administrator/Documents/CHATBOT SOFY/CHATBOT SOFY/'
                            'Gestion humana y administrativa/Cesantias/Produccion/utterances_no_agregadas.xlsx',
               'office': 'C:/Users/Administrator/Documents/CHATBOT SOFY/CHATBOT SOFY/Tecnologia/Outlook/'
                         'Produccion/utterances_no_agregadas.xlsx',
               'enrutador': 'C:/Users/Administrator/Documents/CHATBOT SOFY/CHATBOT SOFY/Enrutador principal/'
                            'Produccion/utterances_no_agregadas.xlsx',
               'programas_gobierno': 'C:/Users/Administrator/Documents/CHATBOT SOFY/Hipotecario/programas de gobierno/' \
                                     'Refinamiento 1/Curacion/utterances_no_agregadas.xlsx'}

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

vBagsMapping = {'office': '#tecnologia', 'cesantias': '#administrar_novedades_de_nomina',
                'programas_gobierno': '#procesos_y_producto',
                'descarte': 'None', 'naturalidad': 'None', 's_n': 'None'}
vThresholdFlag = True

vOmitBags = []
vRouterPredictions = fn_get_router_valPredictions(vMergeTaggedDataFrame, vOmitBags, vBagsMapping, vLogsPath,
                                                  saveFlag=True)

'''---------------------- GENERATE EXCEL FILE TO COMPUTE THE INITIAL EVALUATION OF THE CHILDS -------------'''
vChildTags = ['cesantias', 'office']
vChildsUpdateDict = fn_get_childs_valPredictions(vMergeTaggedDataFrame, vChildTags, vLogsPath, none=True, saveFlag=True)

"---------------------------------  ESTIMATE PERFORMANCE MEASURES -------------------------------------------"

# fn_request_metrics(endpoint=vEndpoint,
#                    pathSave=vPathSave,
#                    configPath=vConfigPath,
#                    dataPath=vDataPredPath,
#                    modelName=vModelName)