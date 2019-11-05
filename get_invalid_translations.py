# Get invalid translations

import json, os, time

SQUAD_PATH = 'D:/Github/trts_crawler/1.1'
SQUAD_FILES = [
    #'dev-v1.1.json',
    'train-v1.1.json'
]

translated_answers = 0
untraslated_answers = 0

for filename in SQUAD_FILES:

    print("Lendo {}...\n".format(filename))

    squadfile_path = os.path.join(SQUAD_PATH, filename.replace('.json', '-traduzido.json'))

    with open(squadfile_path) as json_file:
        json_content = json.load(json_file)

    for i, data_array in enumerate(json_content['data']):

        for j, paragraph in enumerate(data_array['paragraphs']):

            for k, qas_tuple in enumerate(paragraph['qas']):

                answers_to_remove = [answer for answer in qas_tuple['answers'] if answer['answer_start'] != -1]
                answers_to_keep = [answer for answer in qas_tuple['answers'] if answer['answer_start'] == -1]
                translated_answers += len(answers_to_remove)
                untraslated_answers += len(answers_to_keep)

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

    with open(squadfile_path.replace('.json', '-untranslated_only.json'), 'wt') as translated_json:
        json.dump(json_content, translated_json)

doccano_json = {'data': []}

for filename in SQUAD_FILES:

    squadfile_path = os.path.join(SQUAD_PATH, filename.replace('.json', '-traduzido-untranslated_only.json'))

    with open(squadfile_path) as json_file:
        json_content = json.load(json_file)

    for data in json_content['data']:
        doccano_json['data'].append(data)

untranslated_only_doccano_jsonl_set = set()

for data in doccano_json['data']:
    #{"text": "Peter Blackburn", "labels": [ [0, 15, "PERSON"] ]}

    for j, paragraph in enumerate(data['paragraphs']):

        for k, qas_tuple in enumerate(paragraph['qas']):

            for l, answer in enumerate(qas_tuple['answers']):

                new_line = "{{\"text\": \"{}\", \"labels\": [[{}, {}, {}]]}}\n".format(
                    # Parágrafo com resposta original marcada + Pergunta + Resposta + Posição original do início da resposta
                    (
                        paragraph['context'] + 
                        '\\n\\n' +
                        paragraph['context_original'] + 
                        '\\n\\n' +
                        qas_tuple['question'] + 
                        '\\n' + 
                        qas_tuple['question_original'] + 
                        '\\n\\n' + 
                        answer['text'] +
                        '\\n' +
                        answer['text_original']
                    ).replace('\"', '\\"'),
                    int(answer['answer_start_original']),
                    int(answer['answer_start_original']) + len(answer['text']),
                    "\"TRADUCAO\""
                )

                new_line = new_line.replace('\n', '').replace(' \ ', ' \\\\ ')
                new_line += '\n'

                untranslated_only_doccano_jsonl_set.add(new_line)

            '''
            if 'plausible_answers' in qas_tuple:
                for m, plausible_answer in enumerate(qas_tuple['plausible_answers']):
            '''
print("Text {}/{}, paragraph {}/{}, question {}/{}, answer {}/{}".format(
i + 1, len(json_content['data']),
j + 1, len(data_array['paragraphs']),
k + 1, len(paragraph['qas']),
l + 1, len(qas_tuple['answers']), end='\r'))

with open(SQUAD_PATH + '/untranslated_only_doccano_jsonl.jsonl', 'wt', encoding='utf-8') as untranslated_only_doccano_jsonl:

    for line in untranslated_only_doccano_jsonl_set:
        untranslated_only_doccano_jsonl.write(line)