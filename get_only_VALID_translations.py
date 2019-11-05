# Get invalid translations

import json, os, time

SQUAD_PATH = 'D:/Github/trts_crawler/1.1/100_cento'
SQUAD_FILES = [
    #'dev-v1.1.json',
    'train-v1.1.json'
]

translated_answers = 0
untraslated_answers = 0

for filename in SQUAD_FILES:

    print("Lendo {}...\n".format(filename))

    squadfile_path = os.path.join(SQUAD_PATH, filename.replace('.json', '-traduzido-traducao-auto-e-manual.json'))

    with open(squadfile_path) as json_file:
        json_content = json.load(json_file)

    for i, data_array in enumerate(json_content['data']):

        for j, paragraph in enumerate(data_array['paragraphs']):

            for k, qas_tuple in enumerate(paragraph['qas']):

                answers_to_keep = [answer for answer in qas_tuple['answers'] if answer['answer_start'] != -1]
                answers_to_remove = [answer for answer in qas_tuple['answers'] if answer['answer_start'] == -1]
                untraslated_answers += len(answers_to_remove)
                translated_answers += len(answers_to_keep)

                for answer_to_remove in answers_to_remove:
                    qas_tuple['answers'].remove(answer_to_remove)

            qas_tuples_to_remove = [qas for qas in paragraph['qas'] if len(qas['answers']) == 0]

            for qas_to_remove in qas_tuples_to_remove:
                paragraph['qas'].remove(qas_to_remove)

        paragraphs_to_remove = [paragraph for paragraph in data_array['paragraphs'] if len(paragraph['qas']) == 0]

        for paragraph_to_remove in paragraphs_to_remove:
            data_array['paragraphs'].remove(paragraph_to_remove)

    json_data_to_remove = [data for data in json_content['data'] if len(data['paragraphs']) == 0]

    for data_to_remove in json_data_to_remove:
        json_content['data'].remove(data_to_remove)

    print("{} translated answers: {}".format(filename, translated_answers))
    print("{} untranslated answers: {}".format(filename, untraslated_answers))

    with open(squadfile_path.replace('.json', '-apenas_traducoes_corretas.json'), 'wt') as translated_json:
        json.dump(json_content, translated_json)