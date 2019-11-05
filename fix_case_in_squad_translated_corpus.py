import os, json

#SQUAD_PATH = 'D:/Github/trts_crawler/1.1'
SQUAD_PATH = 'D:/Github/trts_crawler/1.1/100_cento'
SQUAD_FILES = [
    #'dev-v1.1-traduzido-traducao-auto-e-manual.json',
    #'train-v1.1-traduzido-apenas_traduzidos_automatico.json'
    'train-v1.1-traduzido-traducao-auto-e-manual-apenas_traducoes_corretas-ideado.json'
]

qntd_corrigidos = []

for filename in SQUAD_FILES:

    squadfile_path = os.path.join(SQUAD_PATH, filename)

    resultado_correcao = {
        'nome': filename,
        'corrigidos': 0,
        'sem_corrigir': 0
    }

    with open(squadfile_path) as squadfile_file:
        squadfile_file = json.load(squadfile_file)

    for i, data_array in enumerate(squadfile_file['data']):

        for j, paragraph in enumerate(data_array['paragraphs']):

            for k, qas_tuple in enumerate(paragraph['qas']):

                for l, answer in enumerate(qas_tuple['answers']):

                    #qntd_corretos += 1

                    assert answer['answer_start'] != -1, "A resposta ainda não foi previamente preenchida!"

                    if (answer['text'].lower() == paragraph['context'][answer['answer_start']:answer['answer_start'] + len(answer['text'])].lower() and answer['text'] != paragraph['context'][answer['answer_start']:answer['answer_start'] + len(answer['text'])]):

                        resultado_correcao['corrigidos'] += 1

                        answer['text'] = paragraph['context'][answer['answer_start']:answer['answer_start'] + len(answer['text'])]
                    else:
                        resultado_correcao['sem_corrigir'] += 1


                    #try:
                    assert answer['text'] in paragraph['context'] or answer['text'] in paragraph['context'], "A resposta não está presente no parágrafo!"
                    #except Exception as exce:
                    #    lista_errados_nao_existente.append({
                    #        "paragraph": paragraph,
                    #        "qas_tuple": qas_tuple,
                    #        "answer": answer
                    #    })

                    #try:
                    assert answer['text'] == paragraph['context'][answer['answer_start']:answer['answer_start'] + len(answer['text'])], "A resposta não bate com os índices de parágrafo selecionados"
                    #except Exception as exce:
                    #    lista_errados_indice_nao_existente.append({
                    #        "paragraph": paragraph,
                    #        "qas_tuple": qas_tuple,
                    #        "answer": answer
                    #    })
                #else:
                #    qntd_errados += 1

    with open(squadfile_path.replace('.json', '-corrigido_case.json'), 'wt') as final_json:
        json.dump(squadfile_file, final_json)

    qntd_corrigidos.append(resultado_correcao)

print("Quantidade corrigidos: {}".format(qntd_corrigidos))