import os, json, torch
from transformers import BertTokenizer, BertForQuestionAnswering

SQUAD_PATHS = [
    '/home/corpora/squad_v11_pt_br/train-v1.1-traduzido-traducao-auto-e-manual-apenas_traducoes_corretas-ideado-corrigido_case_100_cento.json',
    '/home/corpora/squad_v11_pt_br/dev-v1.1-traduzido-traducao-auto-e-manual-corrigido_case.json'
]
MODEL_PATH = 'D:\\Github\\trts_crawler\\1.1\\corpus server\\trained_benchmark_case_100_cento'


tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)

# Colocando o modelo em modo de evaluation
model.eval()
model.to('cuda')

count_size = []

for path in SQUAD_PATHS:
    with open(path) as squadfile_file:
        squadfile_file = json.load(squadfile_file)
    for i, data_array in enumerate(squadfile_file['data']):
        for j, paragraph in enumerate(data_array['paragraphs']):
            for k, qas_tuple in enumerate(paragraph['qas']):
                    count_size.append({
                        'question': qas_tuple['question'],
                        'text': paragraph['context'],
                        'len': len(tokenizer.encode(f"[CLS] {qas_tuple['question']} [SEP] {paragraph['context']} [SEP]"))}
                    )



