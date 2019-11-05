import json, os, uuid

#SQUAD_PATH = 'D:/Github/trts_crawler/1.1'
SQUAD_PATH = 'D:/Github/trts_crawler/1.1/100_cento'
SQUAD_FILES = [
    #'dev-v1.1.json',
    'train-v1.1.json'
]

for squad_json in SQUAD_FILES:
    original_squad_file_path = os.path.join(SQUAD_PATH, squad_json)
    final_squad_file_path = original_squad_file_path.replace('.json', '-traduzido-traducao-auto-e-manual-apenas_traducoes_corretas.json')
    uuids_set = set()

    with open(final_squad_file_path) as final_squad_file:
        final_squad_file = json.load(final_squad_file)

    '''
    with open(original_squad_file_path) as original_squad_file:
        original_squad_file = json.load(original_squad_file)
        original_rows = []

        for i, data_array in enumerate(original_squad_file['data']):
            for j, paragraph in enumerate(data_array['paragraphs']):
                for k, qas_tuple in enumerate(paragraph['qas']):

                    original_rows.append({
                        'context_original': paragraph['context'],
                        'question_original': qas_tuple['question'],
                        'id': qas_tuple['id']
                    })
    '''

    ids = 0

    for i, data_array in enumerate(final_squad_file['data']):

        for j, paragraph in enumerate(data_array['paragraphs']):

            for k, qas_tuple in enumerate(paragraph['qas']):

                if 'id' not in qas_tuple:

                    new_id = str(uuid.uuid4())

                    while new_id in uuids_set:
                        new_id = str(uuid.uuid4())

                    uuids_set.add(new_id)

                    qas_tuple['id'] = new_id

                    ids += 1

                else:

                    uuids_set.add(qas_tuple['id'])

                '''
                doccano_translated_array = [doccano_object for doccano_object in original_rows if (
                    paragraph['context_original'] == doccano_object['context_original'] and 
                    qas_tuple['question_original'] == doccano_object['question_original'] and 
                    doccano_object['id'] != None
                )]

                if len(doccano_translated_array) > 1: doccano_translated_array = [doccano_translated_array[0]]

                assert len(doccano_translated_array) == 1, "ERRO!"
                #doccano_translated_array[0]['id'] = None
                '''

    print("{} ids".format(ids))

with open(final_squad_file_path.replace('.json', '-ideado.json'), 'wt') as final_json:
    json.dump(final_squad_file, final_json)