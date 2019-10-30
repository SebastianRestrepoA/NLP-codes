import pandas as pd

vLogsPath = 'C:/Users/Administrator/Documents/CHATBOT SOFY/ONGOING TRAINING/Refinamiento_5/logs_conversacionales.xlsx'
logs = pd.read_excel(vLogsPath)

conversations = logs['ConversationGuid'].unique()

efectivity = []
interactions_per_conversation = []
for conversationId in conversations:

    conversation_interactions = logs[['IdConversation', 'Date', 'Utterance',
                                      'IntentRecognized', 'BotAnswer']][logs['ConversationGuid'] == conversationId]

    conversation_interactions2 = conversation_interactions[conversation_interactions['IntentRecognized']
                                                           != '#administrar_novedades_de_nomina']

    conversation_interactions3 = conversation_interactions2[conversation_interactions['IntentRecognized']
                                                            != '#tecnologia']

    conversation_interactions4 = conversation_interactions3[conversation_interactions['IntentRecognized']
                                                            != '#procesos_y_producto'].reset_index(drop=True)

    interactions_per_conversation.append(len(conversation_interactions4['Utterance']))

    for idx in range(0, len(conversation_interactions4['Utterance'])-1):

        flag = False

        if conversation_interactions4['Utterance'].iloc[idx+1].lower().strip() == 'si' or conversation_interactions4['Utterance'].iloc[idx+1].lower().strip() == 'no':

            if conversation_interactions4['Utterance'].iloc[idx].lower().strip() == 'si' or conversation_interactions4['Utterance'].iloc[idx].lower().strip() == 'no':
                flag = False
            else:
                flag = True

        if flag is True:
            efectivity.append(conversation_interactions4['Utterance'].iloc[idx+1].lower().strip())

interactions_per_conversation = pd.DataFrame(interactions_per_conversation, columns=['total_of_interactions'])
total_interactions = interactions_per_conversation.sum()
total_of_conversations = len(interactions_per_conversation)

interactions_below_ten = interactions_per_conversation < 10

percentage_of_conversations_with_interactions_lower_than_ten = (len(interactions_per_conversation[interactions_below_ten].dropna())*100)/total_of_conversations

efectivity = pd.DataFrame(efectivity, columns=['Answer'])
yes_answers = efectivity == 'si'

efectivity_rate = (len(efectivity[yes_answers].dropna())*100)/len(efectivity)
