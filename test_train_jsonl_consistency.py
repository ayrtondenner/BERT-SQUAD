import json, os, time

TRAIN_SQUAD_PATH = 'D:\\Github\\trts_crawler\\1.1\\train-v1.1-doccano-manual-translated.json1'

#Teste do train_v1_1.json1
doccano_manual_translations = []
with open(TRAIN_SQUAD_PATH, encoding='utf-8') as manual_translated_squad_doccano_checked:
    for jline in manual_translated_squad_doccano_checked.read().split('\n'):
        if not jline:
            continue

        doccano_object = json.loads(jline)

        if jline and 'annotation_approver' in doccano_object and doccano_object['annotation_approver'] == 'admin' and len(doccano_object['labels']) > 0:

            squad_texts = doccano_object['text'].replace('\n\n', '\n').split('\n')

            assert len(squad_texts) == 6, "O tamanho do string array exportado do Doccano é diferente de 6!"

            assert len(doccano_object['labels']) == 1, "A resposta não apresenta apenas 1 label!"
            assert doccano_object['labels'][0][-1] == 'TRADUCAO', "A resposta tem uma tag diferente de Tradução!"
            
            assert len(doccano_object['labels'][0]) == 3, "A primeira anotação não apresenta 3 valores!"
            assert doccano_object['labels'][0][0] < doccano_object['labels'][0][1], "Os valores de anotações estão errados!"

            squad_object = {
                'context': squad_texts[0],
                'context_original':  squad_texts[1],
                'question': squad_texts[2],
                'question_original': squad_texts[3],
                'text': squad_texts[4],
                'text_original': squad_texts[5],
                'answer_start': doccano_object['labels'][0][0],
                'answer_end': doccano_object['labels'][0][1]
            }

            assert len(squad_object['context']) >= squad_object['answer_start'], "O início da resposta é maior que o tamanho do texto!"
            assert len(squad_object['context']) >= squad_object['answer_end'], "O fim da resposta é maior que o tamanho do texto!"

            squad_object['text'] = squad_object['context'][squad_object['answer_start']:squad_object['answer_end']]

            if squad_object['text'].startswith(' ') or squad_object['text'].endswith(' '):
                if squad_object['text'].startswith(' '):
                    squad_object['answer_start'] += 1

                if squad_object['text'].endswith(' '):
                    squad_object['answer_end'] -= 1

                text_striped = squad_object['text'].strip()
                print('\"{}\" -> \"{}\"'.format(squad_object['text'], text_striped))
                squad_object['text'] = text_striped

            assert squad_object['text'] in squad_object['context'], "A frase resposta não está contida no parágrafo!"
            assert len(squad_object['text']) == squad_object['answer_end'] - squad_object['answer_start'], "O tamanho da resposta escolhida e dos índices da resposta estão diferentes!"
            assert squad_object['text'] == squad_object['context'][squad_object['answer_start']:squad_object['answer_end']], "O texto não é o mesmo delimitado pelos índices da resposta!"

            doccano_manual_translations.append(squad_object)

test_dict = {
    'data': []
}

for item in doccano_manual_translations:
    #test_dict['data'].append(doccano_manual_translations)
    test_dict['data'].append({
        'paragraphs': [{
            'context': item['context'],
            'context_original': item['context_original'],
            'qas': [{
                'question': item['question'],
                'question_original': item['question_original'],
                'answers': [{
                    'answer_start': item['answer_start'],
                    'text': item['text'],
                    'text_original': item['text_original']
                }]
            }]
        }]
    })

for i, data_array in enumerate(test_dict['data']):

    #if data_array['title'] != 'Super_Bowl_50':
    #    continue

    for j, paragraph in enumerate(data_array['paragraphs']):

        for k, qas_tuple in enumerate(paragraph['qas']):

            for l, answer in enumerate(qas_tuple['answers']):

                assert answer['answer_start'] != -1, "A resposta ainda não foi previamente preenchida!"
                assert answer['text'] in paragraph['context'] or answer['text'].lower() in paragraph['context'].lower(), "A resposta não está presente no parágrafo!"
                assert answer['text'].lower() == paragraph['context'][answer['answer_start']:answer['answer_start'] + len(answer['text'])].lower(), "A resposta não bate com os índices de parágrafo selecionados"

with open(TRAIN_SQUAD_PATH.replace('-doccano-manual-translated.json1', '-traduzido-traducao-auto-e-manual.json'), 'wt') as final_json:
    json.dump(test_dict, final_json)