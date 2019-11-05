import json, os, re

SQUAD_PATH = 'D:/Github/trts_crawler/1.1/100_cento'
SQUAD_FILES = [
    #'dev-v1.1.json',
    'train-v1.1.json'
]

def convert_doccano_object_to_squad_object(doccano_object):
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

    #assert not squad_object['text'].startswith(' '), 'A resposta escolhida começa com espaço!'
    #assert not squad_object['text'].endswith(' '), 'A resposta escolhida termina com espaço!'

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

    return squad_object

for filename in SQUAD_FILES:

    errados = 0

    #squadfile_auto_translated_path = os.path.join(SQUAD_PATH, filename.replace('.json', '-traduzido.json'))

    #with open(squadfile_auto_translated_path) as squadfile_auto_translated_file:
    #    squadfile_auto_translated_json = json.load(squadfile_auto_translated_file)

    # TODO: Remover "-untranslated_only_doccano_jsonl" e "-untranslated_only_doccano_jsonl-CHECKED" pois não são mais necessários/úteis
    #manual_translated_squad_doccano_checked_path = os.path.join(SQUAD_PATH, filename.replace('.json', '-untranslated_only_doccano_jsonl-CHECKED.jsonl'))




    manual_translated_squad_doccano_checked_path = os.path.join(SQUAD_PATH, filename.replace('.json', '-doccano-manual-translated.json1'))
    #squadfile_auto_translated_path = os.path.join(SQUAD_PATH, filename.replace('.json', '-traduzido-apenas_traduzidos_automatico.json'))
    squadfile_auto_translated_path = os.path.join(SQUAD_PATH, filename.replace('.json', '-traduzido.json'))

    with open(manual_translated_squad_doccano_checked_path, encoding='utf-8') as manual_translated_squad_doccano_checked:
        doccano_manual_translations = [
            convert_doccano_object_to_squad_object(json.loads(jline)) for jline in manual_translated_squad_doccano_checked.read().split('\n') if (
                jline and 
                len(json.loads(jline)['labels']) == 1 and 
                json.loads(jline)['annotation_approver'] == 'admin' and 
                json.loads(jline)['labels'][0][0] != -1
            )
        ]


    with open(squadfile_auto_translated_path) as squadfile_auto_translated_file:
        squadfile_auto_translated_json = json.load(squadfile_auto_translated_file)

    for i, data_array in enumerate(squadfile_auto_translated_json['data']):

        for j, paragraph in enumerate(data_array['paragraphs']):

            #if paragraph['context_original'] == ("BSkyB's direct-to-home satellite service became available in 10 million homes in 2010, Europe's first pay-TV "
            #"platform in to achieve that milestone. Confirming it had reached its target, the broadcaster said its reach into 36% of households in the UK represented an "
            #"audience of more than 25m people. The target was first announced in August 2004, since then an additional 2.4m customers had subscribed to BSkyB's direct-to-home "
            #"service. Media commentators had debated whether the figure could be reached as the growth in subscriber numbers elsewhere in Europe flattened."):

            #    paragraph['context'] = ("O serviço de satélite direto da BSkyB tornou-se disponível em 10 milhões de lares em 2010, a primeira plataforma de TV paga da Europa "
            #        "a atingir esse marco. Confirmando que atingiu o seu alvo, a emissora disse que seu alcance em 36% das famílias no Reino Unido representou uma audiência de "
            #        "mais de 25 milhões de pessoas. A meta foi anunciada pela primeira vez em agosto de 2004, desde então mais 2,4 milhões de clientes assinaram o serviço "
            #        "direto da BSkyB. Os comentaristas da mídia debateram se o número poderia ser alcançado à medida que o crescimento do número de assinantes em "
            #        "outras partes da Europa se acalmasse.")

            #elif paragraph['context_original'] == ("2013 Economics Nobel prize winner Robert J. Shiller said that rising inequality in the United States and elsewhere is the "
            #"most important problem. Increasing inequality harms economic growth. High and persistent unemployment, in which inequality increases, has a negative effect on "
            #"subsequent long-run economic growth. Unemployment can harm growth not only because it is a waste of resources, but also because it generates redistributive "
            #"pressures and subsequent distortions, drives people to poverty, constrains liquidity limiting labor mobility, and erodes self-esteem promoting "
            #"social dislocation, unrest and conflict. Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support "
            #"economic growth."):

            #    paragraph['context'] = ("Robert J. Shiller, vencedor do Nobel de Economia de 2013, disse que o aumento da desigualdade nos Estados Unidos e "
            #        "em outros lugares é o problema mais importante. O aumento da desigualdade prejudica o crescimento econômico. "
            #        "O desemprego alto e persistente, no qual a desigualdade aumenta, tem um efeito negativo no crescimento econômico subseqüente de longo prazo. "
            #        "O desemprego pode prejudicar o crescimento não apenas porque é um desperdício de recursos, "
            #        "mas também porque gera pressões redistributivas e distorções subsequentes, leva as pessoas à pobreza, limita a mobilidade limitadora da mão-de-obra e "
            #        "erode a auto-estima promovendo deslocamento social, inquietação e conflito. As políticas destinadas a controlar o desemprego e, em particular, a reduzir "
            #        "os seus efeitos associados à desigualdade apoiam o crescimento económico.")


            for k, qas_tuple in enumerate(paragraph['qas']):

                for l, answer in enumerate(qas_tuple['answers']):

                    if answer['answer_start'] == -1:

                        doccano_translated_array = [doccano_object for doccano_object in doccano_manual_translations if (
                            paragraph['context'] == doccano_object['context'] and 
                            paragraph['context_original'] == doccano_object['context_original'] and 
                            qas_tuple['question'] == doccano_object['question'] and 
                            qas_tuple['question_original'] == doccano_object['question_original'] and 
                            #answer['text'] == doccano_object['text'] and 
                            answer['text_original'] == doccano_object['text_original'] 
                        )]

                        if len(doccano_translated_array) == 0:
                            doccano_translated_array = [doccano_object for doccano_object in doccano_manual_translations if (
                                #paragraph['context'] == doccano_object['context'] and 
                                paragraph['context_original'] == doccano_object['context_original'] and 
                                qas_tuple['question'] == doccano_object['question'] and 
                                qas_tuple['question_original'] == doccano_object['question_original'] and 
                                #answer['text'] == doccano_object['text'] and 
                                answer['text_original'] == doccano_object['text_original'] 
                            )]

                        if len(doccano_translated_array) == 0:
                            doccano_translated_array = [doccano_object for doccano_object in doccano_manual_translations if (
                                paragraph['context'] == doccano_object['context'] and 
                                #paragraph['context_original'] == doccano_object['context_original'] and 
                                qas_tuple['question'] == doccano_object['question'] and 
                                qas_tuple['question_original'] == doccano_object['question_original'] and 
                                #answer['text'] == doccano_object['text'] and 
                                answer['text_original'] == doccano_object['text_original'] 
                            )]

                        if len(doccano_translated_array) == 0:
                            errados += 1
                            continue

                        #assert len(doccano_translated_array) == 1, "O número de documentos do tipo Doccano igual o SQuAD não traduzido é diferente de 1!"
                        assert len(doccano_translated_array) != 0, "Nenhum documento do tipo Doccano igual ao SQuAD foi encontrado!"

                        if len(doccano_translated_array) > 1:
                            doccano_translated_array = sorted(doccano_translated_array, key=lambda x: abs(x['answer_start'] - answer['answer_start_original']))
                            doccano_translated_object = doccano_translated_array[0]
                        else:
                            doccano_translated_object = doccano_translated_array[0]

                        answer['answer_start'] = doccano_translated_object['answer_start']
                        answer['text'] = doccano_translated_object['text']

                        # Caso o sistema tenha tido algum problema por causa do caractere '\xa0', que é o espaço
                        # As anotações podem ter tido problemas
                        # Por isso, os índices da resposta anotada vai ser atualizada em relação ao arquivo do SQuAD
                        # Pois os textos do arquivo SQuAD e do arquivo doccano estão diferentes por causa de problemas nesse caractere
                        # O que é mais seguro do que alterar o texto de algum dos dois
                        if '\xa0' in paragraph['context'] and answer['text'] != paragraph['context'][answer['answer_start']:answer['answer_start'] + len(answer['text'])]:
                            new_answer_start = sorted([m.start() for m in re.finditer(doccano_translated_object['text'], paragraph['context'])], key=lambda x: abs(x-doccano_translated_object['answer_start']))[0]
                            answer['answer_start'] = new_answer_start

                            assert answer['text'] == paragraph['context'][answer['answer_start']:answer['answer_start'] + len(answer['text'])], "O índice do arquivo com xa0 ainda não foi corrigido corretamente!"

    '''
    if answer['text'] != paragraph['context'][answer['answer_start']:answer['answer_start'] + len(answer['text'])]:
        if answer['text'] == paragraph['context'][answer['answer_start'] + 1:answer['answer_start'] + len(answer['text']) + 1]:

            # Para um caso de ' Santa Clara, Califórni'
            answer['answer_start'] += 1
        else:
            raise Exception("A mudança manual de índices de resposta não funcionou!")
    '''

    lista_errados_nao_existente = []
    lista_errados_indice_nao_existente = []

    qntd_corretos = 0
    qntd_errados = 0

    for i, data_array in enumerate(squadfile_auto_translated_json['data']):

        #if data_array['title'] != 'Super_Bowl_50':
        #    continue

        for j, paragraph in enumerate(data_array['paragraphs']):

            for k, qas_tuple in enumerate(paragraph['qas']):

                for l, answer in enumerate(qas_tuple['answers']):

                    if answer['answer_start'] != -1:
                    #if 'text_original' in answer:

                        qntd_corretos += 1

                    #assert answer['answer_start'] != -1, "A resposta ainda não foi previamente preenchida!"
                    #assert answer['text'] in paragraph['context'] or answer['text'].lower() in paragraph['context'].lower(), "A resposta não está presente no parágrafo!"
                    #assert answer['text'].lower() == paragraph['context'][answer['answer_start']:answer['answer_start'] + len(answer['text'])].lower(), "A resposta não bate com os índices de parágrafo selecionados"
                    else:
                        qntd_errados += 1

    print("{} itens corretos e {} itens errados".format(qntd_corretos, qntd_errados))

    with open(squadfile_auto_translated_path.replace('.json', '-traducao-auto-e-manual.json'), 'wt') as final_json:
        json.dump(squadfile_auto_translated_json, final_json)