import json, os

SQUAD_PATH = 'D:\\Github\\trts_crawler'

SQUAD_FILES = [
    'dev-v2.0.json',
    'train-v2.0.json'
]

matched_cases = []
unmatched_cases = []

for filename in SQUAD_FILES:

    squadfile_path = os.path.join(SQUAD_PATH, filename.replace('.json', '-traduzido.json'))
    #squadfile_path = os.path.join(SQUAD_PATH, filename)

    with open(squadfile_path) as json_file:
        json_content = json.load(json_file)

    for i, data_array in enumerate(json_content['data']):

        for j, paragraph in enumerate(data_array['paragraphs']):

            for k, qas_tuple in enumerate(paragraph['qas']):

                for l, answer in enumerate(qas_tuple['answers']):

                    answer_start = answer['answer_start']

                    if paragraph['context'][answer_start:answer_start + len(answer['text'])] == answer['text']:
                        #print("Answer: {}".format(answer_start))
                        matched_cases.append(answer)
                    else:
                        unmatched_cases.append(answer)

                if 'plausible_answers' in qas_tuple:

                    for m, plausible_answer in enumerate(qas_tuple['plausible_answers']):

                        if paragraph['context'][answer_start:answer_start + len(answer['text'])] == plausible_answer['text']:
                            #print("Plausible Answer: {}".format(answer_start))
                            matched_cases.append(plausible_answer)
                        else:
                            unmatched_cases.append(plausible_answer)



print("Casos encontrados: {}".format(len(matched_cases)))
print("Casos não encontrados: {}".format(len(unmatched_cases)))
print(matched_cases)