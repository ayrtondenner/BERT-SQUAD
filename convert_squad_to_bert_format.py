import json, os, requests, datetime, time
#import nltk    
#nltk.download('all')
from nltk import PunktSentenceTokenizer

SQUAD_PATH = 'D:\\Github\\trts_crawler\\1.1'

SQUAD_FILES = [
    'dev-v1.1.json',
    'train-v1.1.json'
]

paragraph_text = []
tokenizer = PunktSentenceTokenizer()

for filename in SQUAD_FILES:

    #squadfile_path = os.path.join(SQUAD_PATH, filename.replace('.json', '-traduzido.json'))
    squadfile_path = os.path.join(SQUAD_PATH, filename)

    with open(squadfile_path) as json_file:
        json_content = json.load(json_file)

    for i, data_array in enumerate(json_content['data']):

        for j, paragraph in enumerate(data_array['paragraphs']):

            tokenized_paragraphs = tokenizer.tokenize(paragraph['context'])
            paragraph_text.extend([paragraph + '\n' for paragraph in tokenized_paragraphs])

        paragraph_text.append('\n')

with open(SQUAD_PATH + '\\squad_corpus.txt', 'wt', encoding='utf-8') as squad_traduzido_txt:
    for text in paragraph_text:
        squad_traduzido_txt.write(text)