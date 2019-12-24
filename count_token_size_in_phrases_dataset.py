import os, json, torch
from transformers import BertTokenizer, BertForQuestionAnswering

MODEL_PATH = 'D:\\Github\\trts_crawler\\1.1\\corpus server\\trained_benchmark_case_100_cento'

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)

# Colocando o modelo em modo de evaluation
model.eval()
model.to('cuda')

count_size = []
