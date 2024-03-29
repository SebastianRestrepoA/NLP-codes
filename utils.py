import pandas as pd
import string
import json
import datetime
import xlsxwriter
import pickle
import numpy as np
import zipfile
import requests


def metrics_mean(df_results, intent_names, measure):

    if measure == 'all':

        metrics_names = ['Precision', 'Std', 'Recall', 'Std', 'F1-Score', 'Std']
        measures = list(df_results.columns)
        results = np.zeros((len(intent_names), len(measures)*2))

        for idx_int, intent in enumerate(intent_names):
            k = 0
            for measure in measures:
                results[idx_int, k] = np.mean(df_results.loc[intent][measure])
                results[idx_int, k+1] = np.std(df_results.loc[intent][measure])
                k += 2
    if measure == 'f1-score':

        results = np.zeros((len(intent_names), 2))
        metrics_names = ['F1-Score', 'Std']

        for idx_int, intent in enumerate(intent_names):
            results[idx_int, 0] = np.mean(df_results.loc[intent]['f1-score'])
            results[idx_int, 1] = np.std(df_results.loc[intent]['f1-score'])

    return pd.DataFrame(results, index=intent_names, columns=metrics_names)


## ONGOING TRAINING

def fn_select_logs(path, date_to_filter):

    # import datetime
    # date_to_filter = 'YYYY-MM-DD HH:MM'

    df = pd.read_excel(path)
    df['Date'] = df['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    date_time = datetime.datetime.strptime(date_to_filter, '%Y-%m-%d %H:%M')
    logs = df[df['Date'] >= date_time]
    writer = pd.ExcelWriter('logs.xlsx', engine='xlsxwriter')
    logs.to_excel(writer)
    writer.save()


def multiple_dfs_to_excel(df_list, writer, sheets, spaces):
    # writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    row = 0
    for dataframe in df_list:
        dataframe.to_excel(writer, sheet_name=sheets, startrow=row , startcol=0)
        row = row + len(dataframe.index) + spaces + 1


def fn_save_obj(path, listDataFrame):
    with open(path, "wb") as fp:
        pickle.dump(listDataFrame, fp)


def fn_load_obj(path):
    with open(path, "rb") as fp:
        vListDataFrame = pickle.load(fp)
    return vListDataFrame


def fn_save_txt(path, text):
    with open(path, "w") as fp:
        fp.write(text)


def fn_load_json(jsonAppPath):
    with open(jsonAppPath, encoding="utf8") as f:
        return json.load(f)


def fn_get_intents_from_df(vData):
    vIntentsDict = dict()
    for name, group in vData.groupby("real_intent"):
        vIntentsDict[name] = group["utterance"].tolist()
    return vIntentsDict


def fn_add_LuisAnswer_columns(row):

    """ This function organize the logs answers....

    :param row: pandas series with the logs collected from one iteration.

    :return: pandas series with the logs reorganized.
    """
    vJson = json.loads(row['LuisAnswer'])
    row['IntentRecognized_1'] = vJson['intents'][0]['intent']
    row['Score_1'] = vJson['intents'][0]['score']
    row['IntentRecognized_2'] = vJson['intents'][1]['intent']
    row['Score_2'] = vJson['intents'][1]['score']
    row['EntitiesRecognized'] = vJson['entities']
    return row


def replace_breaking_space(row):
    row['Utterance'] = row['Utterance'].replace('\xa0',' ')
    return row


def fn_match_entity_tagged(row):
    row['Entity_Match'] = False
    y = [item.strip().lower() for item in row['@entity_name, value, synonym'].split(',')]
    entityTagged = y[2]
    for entityRecognized in json.loads(row['EntitiesRecognized_x'].replace("'",'"')):
        if entityRecognized['entity'] == entityTagged:
            row['Entity_Match'] = True
    return row


def fn_generate_tagger_file(vLogsPath, appsID, vKnowledgeBasePaths, vNoAddPaths):

    """ This function ....

    :param vLogsPath: string variable with local path where is saved .xlsx file with the logs collected from production.
           appsID: dictionary variable with the IDs of the applications in production.
           vKnowledgeBasePaths: dictionary variable with the paths where are saved the knowledge bases of the
           applications in production.
           vNoAddPaths: dictionary variable with the paths where are saved the utterances that were not add to the
           knowledge bases in production.


    :return:


    """

    if vLogsPath.split('.')[-1] == 'xlsx':
        vLogsDataFrame = pd.read_excel(vLogsPath)
    elif vLogsPath.split('.')[-1] == 'csv':
        vLogsDataFrame = pd.read_csv(vLogsPath)
    else:
        raise Exception('Invalid file extension: .{}'.format(vLogsPath.split('.')[-1]))

    vLogsDataFrame = vLogsDataFrame[vLogsDataFrame['Component'] == 'Luis']
    vLogsDataFrame = vLogsDataFrame.apply(lambda row: fn_add_LuisAnswer_columns(row), axis=1)

    # columnList = ['AppLuis', 'Utterance', 'IntentRecognized', 'Score', 'IntentRecognized_1', 'Score_1',
    #               'IntentRecognized_2', 'Score_2', 'Threshold', 'Entities', 'EntitiesRecognized']
    columnList = ['AppLuis', 'Utterance', 'IntentRecognized_1', 'Score_1',
                  'IntentRecognized_2', 'Score_2', 'Threshold', 'EntitiesRecognized']

    vRouterDataFrame = pd.DataFrame(columns=columnList)
    vChildsDataFrame = pd.DataFrame(columns=columnList)

    for app in appsID:
        vAuxDataFrame = vLogsDataFrame[vLogsDataFrame['AppLuis'] == appsID[app]]
        vAuxDataFrame['Utterance'] = vAuxDataFrame['Utterance'].str.lower().str.strip()
        vAuxDataFrame = vAuxDataFrame.drop_duplicates(subset='Utterance')
        vAuxDataFrame = vAuxDataFrame[columnList]
        vAuxDataFrame['AppLuis'] = app
        if app != 'Enrutador':
            vChildsDataFrame = pd.concat([vChildsDataFrame, vAuxDataFrame])
        else:
            vRouterDataFrame = vAuxDataFrame

    vTaggerDataFrame = vRouterDataFrame.merge(vChildsDataFrame, on='Utterance', how='outer', indicator=True)
    vTaggerDataFrame = vTaggerDataFrame.dropna(axis=0, subset=['Utterance'])

    fn_generate_drop_down_lists(vTaggerDataFrame, vKnowledgeBasePaths, vNoAddPaths, vLogsPath)


def fn_generate_drop_down_lists(df_tagger, vPaths, vNoAddPaths, vLogsPath):

    """ This function generates drop down lists based on the intents of the knowledge bases in production.


    :param df_tagger: pandas dataframe with the utterances collected from production.
    :param vPaths: dictionary variable with the paths where are saved the knowledge bases of the
           applications in production
    :param vNoAddPaths: dictionary variable with the paths where are saved the utterances that were not add to the
           knowledge bases in production.
    :param vLogsPath: string variable e with local path where is saved .xlsx file with the logs collected from
           production

    :returns excel file called tagger with the utterances to be tagged through drop down lists of each domain.
             excel file called tagger hide with the intents recognized by Luis for every utterance collected.
             excel file called merge auto tagger with the intents recognized by Luis for every utterance collected and
             the real intent obtained from the knowledge bases. This file saves the utterances that were collected
             in production and they are in the knowledge base.

    """

    intents_names = {}
    utterances_knowledge_base = []
    for key in vPaths.keys():
        domain = pd.read_excel(vPaths[key])
        intents_names[key] = list(set(domain['Intent'])) + [key]
        # intents_names[key].remove('None')
        utterances_knowledge_base.append(pd.read_excel(vPaths[key]))
        utterances_knowledge_base.append(pd.read_excel(vNoAddPaths[key]))

    # utterances_knowledge_base = pd.concat(utterances_knowledge_base).reset_index(drop=True)
    # # Verificar cuales de tagger estan en la base de conocimiento
    # comparison = df_tagger['Utterance'].isin(utterances_knowledge_base['Utterance'])
    # vRepeatedUtterancesHide = df_tagger[comparison==True]
    # vTaggerNoKnowlegdeBase = df_tagger[comparison==False]
    # # Verificar cuales de la bc estan en tagger
    # idx = utterances_knowledge_base['Utterance'].isin(df_tagger['Utterance'])
    # vAutoTagger = pd.concat([pd.Series(), utterances_knowledge_base[idx], pd.Series()], axis=1)
    # vAutoTagger.columns = ['ASK', 'Dominio', 'Intent', 'Utterance', '@entity_name, value, synonym']
    #
    # vMergeAutoTagger = fn_merge_tagged_formats(vAutoTagger, vRepeatedUtterancesHide)

    vWriter1 = pd.ExcelWriter(vLogsPath[0:vLogsPath.rfind('/') + 1] + 'Tagger_Hide.xlsx')
    # vTaggerNoKnowlegdeBase.to_excel(vWriter1, 'logs', index=False)
    df_tagger.to_excel(vWriter1, 'logs', index=False)
    vWriter1.save()
    #
    # vWriter2 = pd.ExcelWriter(vLogsPath[0:vLogsPath.rfind('/') + 1] + 'Merge_Auto_Tagger.xlsx')
    # vMergeAutoTagger.to_excel(vWriter2, 'logs', index=False)
    # vWriter2.save()

    path = vLogsPath[0:vLogsPath.rfind('/') + 1] + 'Tagger.xlsx'
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()
    worksheet2 = workbook.add_worksheet()

    # Add a format for the header cells.
    header_format = workbook.add_format({
        'border': 1,
        'bg_color': '#C6EFCE',
        'bold': True,
        'text_wrap': True,
        'valign': 'vcenter',
        'indent': 1,
    })

    # Set up layout of the worksheet.
    worksheet.set_column('A:A', 10)
    worksheet.set_column('B:B', 20)
    worksheet.set_column('C:C', 25)
    worksheet.set_column('D:D', 50)
    worksheet.set_column('E:E', 20)

    worksheet.set_row(0, 36)
    worksheet.write('A1', 'ASK', header_format)
    worksheet.write('B1', 'Dominio', header_format)
    worksheet.write('C1', 'Intent', header_format)
    worksheet.write('D1', 'Utterance', header_format)
    worksheet.write('E1', '@entity_name, value, synonym', header_format)

    # utterances = list(vTaggerNoKnowlegdeBase['Utterance'].values)
    utterances = list(df_tagger['Utterance'].values)
    worksheet.write_column('D2', utterances)

    domain_names = list(intents_names.keys()) + ['naturalidad', 'descarte', 's_n']
    columns = list(string.ascii_uppercase)
    columns = columns[4:len(domain_names)+1]
    worksheet2.write_column('A1', domain_names)
    workbook.define_name('dominios', '=Sheet2!$A$1:$A$'+str(len(domain_names)))
    worksheet2.write_column('B1', ['naturalidad'])
    workbook.define_name('naturalidad', '=Sheet2!$B$1')
    worksheet2.write_column('C1', ['descarte'])
    workbook.define_name('descarte', '=Sheet2!$C$1')
    worksheet2.write_column('D1', ['s_n'])
    workbook.define_name('s_n', '=Sheet2!$D$1')

    for column, d in zip(columns, list(intents_names.keys())):
        worksheet2.write_column(column+'1', intents_names[d])
        reference = '=Sheet2!$'+column+'$1:$'+column + '$' + str(len(intents_names[d]))
        workbook.define_name(d, reference)

    for i in range(2, len(utterances) + 2):
        worksheet.data_validation('B' + str(i), {'validate': 'list', 'source': '=dominios'})
        excel_fn = '=INDIRECT($B$' + str(i) + ')'
        worksheet.data_validation('C' + str(i), {'validate': 'list', 'source': excel_fn})

    workbook.close()


def fn_merge_tagged_formats(vTagged, vTaggedHide):

    vTagged = vTagged.apply(lambda row: replace_breaking_space(row), axis=1)
    vTaggedHide = vTaggedHide.apply(lambda row: replace_breaking_space(row), axis=1)

    vOuterMerge = vTagged.merge(vTaggedHide, on='Utterance', how='outer', indicator='_merge_2')
    print("Utterances que no coinciden:\n", vOuterMerge[vOuterMerge['_merge_2'] != 'both']['Utterance'].to_string())

    return vOuterMerge[vOuterMerge['_merge_2'] == 'both']


def fn_get_router_valPredictions(mergeTaggedDataFrame, vOmitBags, bagsMapping, vPath, saveFlag=False):

    vRouter = mergeTaggedDataFrame[(mergeTaggedDataFrame['_merge'] != 'right_only') &
                                   -(mergeTaggedDataFrame['Dominio'].isin(vOmitBags))]
    vRouter = vRouter.replace({"Dominio": bagsMapping})
    print('Dominios:', vRouter['Dominio'].unique())
    vRouter = vRouter.apply(lambda row: fn_map_router_predictions(row), axis=1)
    vRouter = vRouter.rename(index=str, columns={"Utterance": "utterance", "IntentRecognized_1_x": "pred_intent",
                                                 "Score_1_x": "score", "Threshold_x": "threshold"})
    vRouter = vRouter[['utterance', 'real_intent', 'pred_intent', 'score', 'threshold']]

    if saveFlag:
        writer = pd.ExcelWriter(vPath[0:vPath.rfind('/') + 1] + 'Router_predictions.xlsx', engine='xlsxwriter')
        vRouter.to_excel(writer, index=False)
        writer.save()

    return


def fn_map_router_predictions(row):
    if row['Dominio']=='enrutador':
        row['real_intent'] = row['Intent'] if row['Intent']!='enrutador' else 'None'
    else:
        row['real_intent'] = row['Dominio']
    return row


def fn_get_childs_valPredictions(mergeTaggedDataFrame, childTags, vPath, none=True, saveFlag=False):
    vChildsUpdateDict = {c: '' for c in childTags}

    for child in childTags:
        vChild = mergeTaggedDataFrame[(mergeTaggedDataFrame['AppLuis_y'].str.lower() == child) &
                                      (mergeTaggedDataFrame['_merge'] != 'left_only')]
        vChild = vChild.apply(lambda row: fn_map_child_predictions(row), axis=1)
        vChild = vChild.rename(index=str, columns={"Utterance": "utterance",
        "IntentRecognized_1_y": "pred_intent",
        "Score_1_y": "score",
        "Threshold_y": "threshold"})
        if not none: vChild = vChild[vChild['real_intent'] != 'None']
        vChildsUpdateDict[child] = vChild[['utterance', 'real_intent', 'pred_intent', 'score', 'threshold']]

        if saveFlag:
            writer = pd.ExcelWriter(vPath[0:vPath.rfind('/') + 1] + child + '_predictions.xlsx', engine='xlsxwriter')
            vChildsUpdateDict[child].to_excel(writer, index=False)
            writer.save()

    return vChildsUpdateDict


def fn_get_orphan_childs(mergeTaggedDataFrame, childTags):
    vOrphans = mergeTaggedDataFrame[(mergeTaggedDataFrame['_merge'] == 'left_only') & (mergeTaggedDataFrame['Dominio']
                                                                                       .isin(childTags))]
    vOrphans = vOrphans.apply(lambda row: fn_map_child_predictions(row), axis=1)
    vOrphans = vOrphans.rename(index=str, columns={"Utterance": "utterance", "IntentRecognized_1_x": "pred_intent",
                                                   "Score_1_x": "score", "Threshold_x": "threshold"})

    return vOrphans[['Dominio', 'utterance', 'real_intent', 'pred_intent', 'score', 'threshold']]


def fn_map_child_predictions(row):
    if (row['Intent'].lower() == row['Dominio'].lower()) or (row['Dominio'].lower() != row['AppLuis_y'].lower()):
        row['real_intent'] = 'None'
    else:
        row['real_intent'] = row['Intent']
    return row


def fn_unzip_file(path_to_zip_file, directory_to_extract_to):

    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()


# COGNITIVE ARCHITECTURE VALIDATION
def fn_luis_response(url, key, utterance):

    headers = {
        # Request headers
        'Ocp-Apim-Subscription-Key': key,
    }

    params = {
        # Query parameter
        'q': utterance,
        # Optional request parameters, set to default values
        'timezoneOffset': '0',
        'verbose': 'true',
        'spellCheck': 'false',
        'staging': 'false',
    }

    entity_recognition = []
    intent_prediction = []

    try:
        r = requests.get(url, headers=headers, params=params)
        output = r.json()
        intent_prediction = output['topScoringIntent']

        for i in range(0, len(output['entities'])):
            entity_recognition.append(output['entities'][i]['resolution']['values'][0])

        # print(prediction)

    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

    luis_output = {'intent_recognition': intent_prediction, 'entities_recognition': entity_recognition}

    return luis_output


def fn_sofy_response(vEndPoints, autoring_key, utterance):

    vMainRouterOutput = fn_luis_response(vEndPoints['MainRouter'], autoring_key['MainRouter'], utterance)
    utterance_path = [pd.DataFrame.from_dict(vMainRouterOutput['intent_recognition'], orient='index').
                          rename(index={'intent': 'main_router_output', 'score': 'score_main_router'}).T]

    if vMainRouterOutput['intent_recognition']['intent'] == 'gestion_humana':

        vFinalOutput = fn_luis_response(vEndPoints['Cesantias'], autoring_key['Cesantias'], utterance)

    elif vMainRouterOutput['intent_recognition']['intent'] == '#tecnologia':

        vFinalOutput = fn_luis_response(vEndPoints['Office'], autoring_key['Office'], utterance)

    elif vMainRouterOutput['intent_recognition']['intent'] == '#procesos_y_producto':

        vHipotecarioRouterOutput = fn_luis_response(vEndPoints['HipotecarioRouter'], autoring_key['HipotecarioRouter'], utterance)
        utterance_path.append(pd.DataFrame.from_dict(vHipotecarioRouterOutput['intent_recognition'], orient='index').
                              rename(index={'intent': 'hip_router_output', 'score': 'score_hip_router'}).T)

        if vHipotecarioRouterOutput['intent_recognition']['intent'] == '#hip_programas_de_gobierno':
            vFinalOutput = fn_luis_response(vEndPoints['ProgramasGobierno'], autoring_key['ProgramasGobierno'], utterance)

        elif vHipotecarioRouterOutput['intent_recognition']['intent'] == 'None':

            vFinalOutput = vHipotecarioRouterOutput

    elif vMainRouterOutput['intent_recognition']['intent'] == '#saludar':

        vFinalOutput = vMainRouterOutput

    elif vMainRouterOutput['intent_recognition']['intent'] == '#despedir':

        vFinalOutput = vMainRouterOutput

    elif vMainRouterOutput['intent_recognition']['intent'] == '#preguntar':

        vFinalOutput = vMainRouterOutput

    elif vMainRouterOutput['intent_recognition']['intent'] == '#calificarproceso':

        vFinalOutput = vMainRouterOutput

    elif vMainRouterOutput['intent_recognition']['intent'] == 'None':

        vFinalOutput = vMainRouterOutput

    utterance_path.append(pd.DataFrame.from_dict(vFinalOutput['intent_recognition'], orient='index').
                          rename(index={'intent': 'final_output', 'score': 'score_final_output'}).T)

    print(utterance + ' ----->  ' + vFinalOutput['intent_recognition']['intent'])
    return pd.concat(utterance_path, axis=1)


def fn_entities_sofy_response(vEndPoints, autoring_key, utterance):

    try:

        main_router_decision = pd.DataFrame([utterance, None, None, None],
                                            index=['utterance', 'main_entity',
                                                   'main_intent', 'main_score'])

        sub_router_decision = pd.DataFrame([None, None, None],
                                           index=['hip_router_entity', 'hip_router_intent', 'hip_router_score'])

        # final_decision = pd.DataFrame([None, None], index=['final_intent', 'final_score'])

        vMainRouterOutput = fn_luis_response(vEndPoints['MainRouter'], autoring_key['MainRouter'], utterance)

        entities_recognition = list(set(vMainRouterOutput['entities_recognition']))
        if 'no' in entities_recognition: entities_recognition.remove('no')
        if 'si' in entities_recognition: entities_recognition.remove('si')

        if len(entities_recognition) == 1:

            main_router_decision = pd.DataFrame([utterance, entities_recognition[0], None, None],
                                                index=['utterance', 'main_entity',
                                                       'main_intent', 'main_score'])

        elif len(entities_recognition) != 1:

            main_router_decision = pd.DataFrame([utterance, None, vMainRouterOutput['intent_recognition']['intent'],
                                                 vMainRouterOutput['intent_recognition']['score']],
                                                index=['utterance', 'main_entity', 'main_intent', 'main_score'])

        if ('gestion_humana' in entities_recognition and len(entities_recognition) == 1) or\
                (vMainRouterOutput['intent_recognition']['intent'] == '#administrar_novedades_de_nomina' and
                 len(entities_recognition) != 1):

            vGHRouterOutput = fn_luis_response(vEndPoints['GhRouter'], autoring_key['GhRouter'], utterance)

            GH_entities_recognition = list(set(vGHRouterOutput['entities_recognition']))

            if len(GH_entities_recognition) == 1:

                sub_router_decision = pd.DataFrame([GH_entities_recognition[0], None, None],
                                                   index=['GH_router_entity', 'GH_router_intent', 'GH_router_score'])

            elif len(GH_entities_recognition) != 1:

                sub_router_decision = pd.DataFrame([None, vGHRouterOutput['intent_recognition']['intent'],
                                                    vGHRouterOutput['intent_recognition']['score']],
                                                   index=['GH_router_entity', 'GH_router_intent', 'GH_router_score'])

            if ('cesantias' in GH_entities_recognition and len(GH_entities_recognition) == 1) or \
                    (vGHRouterOutput['intent_recognition']['intent'] == '#nom_cesantias' and
                     len(GH_entities_recognition) != 1):

                vFinalOutput = fn_luis_response(vEndPoints['Cesantias'], autoring_key['Cesantias'], utterance)

            if ('vacaciones' in GH_entities_recognition and len(GH_entities_recognition) == 1) or \
                    (vGHRouterOutput['intent_recognition']['intent'] == '#nom_vacaciones' and
                     len(GH_entities_recognition) != 1):

                vFinalOutput = fn_luis_response(vEndPoints['Vacaciones'], autoring_key['Vacaciones'], utterance)

            elif (vGHRouterOutput['intent_recognition']['intent'] == 'None' and
                  len(GH_entities_recognition) != 1):

                vFinalOutput = vGHRouterOutput

        elif ('producto' in entities_recognition and len(entities_recognition) == 1) or\
                (vMainRouterOutput['intent_recognition']['intent'] == '#producto' and
                 len(entities_recognition) != 1):

            vHipotecarioRouterOutput = fn_luis_response(vEndPoints['HipotecarioRouter'],
                                                        autoring_key['HipotecarioRouter'],
                                                        utterance)

            hip_entities_recognition = list(set(vHipotecarioRouterOutput['entities_recognition']))

            if len(hip_entities_recognition) == 1:

                sub_router_decision = pd.DataFrame([hip_entities_recognition[0], None, None],
                                                   index=['hip_router_entity', 'hip_router_intent', 'hip_router_score'])

            elif len(hip_entities_recognition) != 1:

                sub_router_decision = pd.DataFrame([None, vHipotecarioRouterOutput['intent_recognition']['intent'],
                                                    vHipotecarioRouterOutput['intent_recognition']['score']],
                                                   index=['hip_router_entity', 'hip_router_intent', 'hip_router_score'])

            # utterance_path.append(pd.DataFrame.from_dict(vHipotecarioRouterOutput, orient='index').
            #                       rename(index={'intent': 'hip_router_output', 'score': 'score_hip_router'}).T)

            if ('programas_de_gobierno' in hip_entities_recognition and len(hip_entities_recognition) == 1) or \
                    (vHipotecarioRouterOutput['intent_recognition']['intent'] == '#hip_programas_de_gobierno' and
                     len(hip_entities_recognition) != 1):

                vFinalOutput = fn_luis_response(vEndPoints['ProgramasGobierno'], autoring_key['ProgramasGobierno'],
                                                utterance)

            elif ('compra_de_cartera' in hip_entities_recognition and len(hip_entities_recognition) == 1) or \
                    (vHipotecarioRouterOutput['intent_recognition']['intent'] == '#hip_compra_de_cartera' and
                     len(hip_entities_recognition) != 1):

                vFinalOutput = fn_luis_response(vEndPoints['CompraCartera'], autoring_key['CompraCartera'],
                                                utterance)

            elif (vHipotecarioRouterOutput['intent_recognition']['intent'] == 'None' and
                  len(hip_entities_recognition) != 1):

                vFinalOutput = vHipotecarioRouterOutput

        elif vMainRouterOutput['intent_recognition']['intent'] == '#tecnologia':

            vFinalOutput = fn_luis_response(vEndPoints['Office'], autoring_key['Office'], utterance)

        elif vMainRouterOutput['intent_recognition']['intent'] == '#saludar':

            vFinalOutput = vMainRouterOutput

        elif vMainRouterOutput['intent_recognition']['intent'] == '#despedir':

            vFinalOutput = vMainRouterOutput

        elif vMainRouterOutput['intent_recognition']['intent'] == '#preguntar':

            vFinalOutput = vMainRouterOutput

        elif vMainRouterOutput['intent_recognition']['intent'] == '#calificarproceso':

            vFinalOutput = vMainRouterOutput

        elif vMainRouterOutput['intent_recognition']['intent'] == 'None':

            vFinalOutput = vMainRouterOutput

        final_decision = pd.DataFrame([vFinalOutput['intent_recognition']['intent'],
                                        vFinalOutput['intent_recognition']['score']],
                                      index=['final_intent', 'final_score'])

        print(utterance + ' ----->  ' + vFinalOutput['intent_recognition']['intent'])

        path = pd.concat((main_router_decision.T, sub_router_decision.T, final_decision.T), axis=1)

    except:
        print("La utterance: " + utterance + 'ha presentado problemas')

    return path


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
