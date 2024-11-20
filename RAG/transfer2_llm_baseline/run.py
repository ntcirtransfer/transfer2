import json
import os

import torch
from tqdm import tqdm

from qa_llm import QA_LLM

def generate_predictions(jsonl_file):
    llm = QA_LLM()
    llm.load_model()
    llm.set_default_sys_prompt("""あなたはクイズ番組の回答者です。queryに20文字以内で答えだけを単語で簡潔に回答してください。\n""")

    with open(jsonl_file,encoding='utf8') as f:
        jsonl_data = [json.loads(l) for l in f.readlines()]

    with torch.no_grad():
        res_data = []
        for d in tqdm(jsonl_data):
            r = {'qid': d['qid'], 'question': d['question']}

            text = f"""Query: {d['question']}\n
            """
            r['prediction'] = llm.generate_output(text,after_inst=' ### Answer:').split('\n')[0].replace('###','')
            print(r['prediction'])
            res_data.append(r)

    with open(f'{results_dir}/result_withoutRAG.jsonl','wt') as fout:
        fout.write('\n'.join([json.dumps(l,ensure_ascii=False) for l in res_data])+'\n')


def generate_predictions_RAG(dpr_file,final_rank=30):
    llm = QA_LLM()
    llm.load_model()
    llm.set_default_sys_prompt("""あなたはクイズ番組の回答者です。
            コンテキスト情報を参考にして、queryに20文字以内で答えだけを単語で簡潔に回答してください。\n""")

                                          
    with torch.no_grad():
        with open(dpr_file,encoding='utf-8',mode='rt') as f:
            jsonl_data = [json.loads(l) for l in f.readlines()]

        for rank in range(0,final_rank):
            res_data = []
            print(f'rank {rank}')
            for d in tqdm(jsonl_data):
                r = {'qid': d['id'], 'question': d['question']}
                r['context'] = d['ctxs'][rank]['text']

                text = f"""Context: {r['context']}\n
                        Query: {d['question']}\n
                """
                r['prediction'] = llm.generate_output(text,after_inst=' ### Answer:').split('\n')[0].replace('###','')
                print(r['prediction'])
                res_data.append(r)

            with open(f'{results_dir}/result_rank{rank}.jsonl','wt') as fout:
                fout.write('\n'.join([json.dumps(l,ensure_ascii=False) for l in res_data])+'\n')
    

if __name__ == '__main__':

    results_dir = 'results'
    os.makedirs(results_dir,exist_ok=True)

    generate_predictions('datasets/aio_02_dev_v1.0.jsonl')
    generate_predictions_RAG('DprRetrieved/dev.jsonl')

