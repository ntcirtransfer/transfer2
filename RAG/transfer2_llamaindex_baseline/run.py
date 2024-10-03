import logging
import os
from tqdm import tqdm

import json

from transformers import AutoTokenizer

from llama_index.service_context import ServiceContext, set_global_service_context
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings import (
    OptimumEmbedding,
    LangchainEmbedding
)
from llama_index.node_parser import SentenceSplitter
from llama_index.storage import StorageContext
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage
)

# cf https://docs.llamaindex.ai/en/stable/optimizing/basic_strategies/basic_strategies/
# cf https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/
# cf https://note.com/npaka/n/nbe5f9849b723

from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Any, List

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)


# add "query" to HuggingFaceEmbeddings class
class HuggingFaceQueryEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(["query: " + text for text in texts])

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query("query: " + text)


llm_model_name = 'elyza/ELYZA-japanese-Llama-2-7b-fast-instruct'
#llm_model_name = 'elyza/Llama-3-ELYZA-JP-8B'
#llm_model_name = 'tokyotech-llm/Swallow-7b-instruct-hf'

#embed_model_name = 'intfloat/multilingual-e5-large'  ## it takes too much CPU memory (80GB is insufficient for chunk_size=2048)
embed_model_name = 'intfloat/multilingual-e5-small'

hybrid = False  # dense/sparse mixed search

if 'Llama-3' in llm_model_name:
    is_llama3 = True
else:
    is_llama3 = False

index = None
chunk_size = 2048

if hybrid:
    from llama_index.vector_stores import QdrantVectorStore
    from qdrant_client import QdrantClient
    storage_context_dir = "./storage_context_{0}_hybrid".format(chunk_size)
else:
    storage_context_dir = "./storage_context_{0}".format(chunk_size)


def prepare_models():
    print('## prepare for embedding model')
    #embed_model = OptimumEmbedding(folder_name="./sonoisa_sbert_v2")

    embed_model = LangchainEmbedding(
        HuggingFaceQueryEmbeddings(model_name=embed_model_name)
    )

    #embeddings = embed_model.get_text_embedding("Hello World!")
    #print(len(embeddings))
    #print(embeddings[:5])

    print('## prepare for LLM')
    llm_tokenizer=AutoTokenizer.from_pretrained(llm_model_name)
    llm = HuggingFaceLLM(
        model_name=llm_model_name,
        tokenizer=llm_tokenizer,
        max_new_tokens=256,
        device_map="auto",
        generate_kwargs={'pad_token_id': llm_tokenizer.eos_token_id} if is_llama3 else {},
    )

    print('## set context')
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=20,
            paragraph_separator="\n\n",
            tokenizer=AutoTokenizer.from_pretrained(embed_model_name)
        )
    )
    set_global_service_context(service_context)

    if hybrid:
        # creates a persistant index to disk
        client = QdrantClient(path=storage_context_dir)

        # create our vector store with hybrid indexing enabled
        # batch_size controls how many nodes are encoded with sparse vectors at once
        vector_store = QdrantVectorStore(
            "AIO", client=client, enable_hybrid=True, batch_size=20
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    else:
        storage_context = None

    return service_context, storage_context


def make_embeddings(storage_context):

    print('## step 2/3 making embeddings')
    documents = SimpleDirectoryReader("data").load_data()

    if hybrid:
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress = True
        )
    else:
        index = VectorStoreIndex.from_documents(documents,show_progress=True)

    os.makedirs(storage_context_dir,exist_ok=True)
    storage_context = index.storage_context
    storage_context.persist(persist_dir=storage_context_dir)

    def retrieve_ex():
        for r in index.as_retriever(similarity_top_k=10).retrieve('アンパサンドの由来は?'):
            print(r.id_)
            print(r.get_score())
            print(r.get_content().replace("\n","")[:100])
            print("-----")
    
    retrieve_ex()

    return index


def generate_answers(questions):
    
    print('## step 3/3 generating answers')
    top_k = 1
    top_k_sparse = 1

    from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt

    # generate only answer within 20 characters for a question
    if is_llama3:
        qa_template = QuestionAnswerPrompt("""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        あなたはクイズ番組の回答者です。質問に20文字以内で答えだけを単語で簡潔に回答してください。
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        以下にコンテキスト情報を提供します。\n
        {context_str}
        \n
        質問: {query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>回答: """)

        refine_template = RefinePrompt("""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        あなたは、既存の回答を改良する際に3つのモードで厳密に動作するクイズ番組の回答者です。
        1. 新しいコンテキストを使用して元の回答を**書き直す**。\n
        2. 新しいコンテキストが役に立たない場合は、元の回答を**繰り返す**。\n
        3. 質問に20文字以内で**答えだけ**を単語で簡潔に回答する。\n
        回答内で元の回答やコンテキストを直接参照しないでください。\n
        疑問がある場合は、元の答えを繰り返してください。
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        New Context: {context_msg}\n
        Query: {query_str}\n
        Original Answer: {existing_answer}\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        New Answer: """)
    else:
        qa_template = QuestionAnswerPrompt("""<s>[INST] <<SYS>>
        あなたはクイズ番組の回答者です。
        <</SYS>>
        以下にコンテキスト情報を提供します。\n
        {context_str}
        \n
        queryに20文字以内で答えだけを単語で簡潔に回答してください。\n
        Query: {query_str}\n
        Answer: 
        [/INST]
        """)
        refine_template = RefinePrompt("""<s>[INST] <<SYS>>
        あなたは、既存の回答を改良する際に3つのモードで厳密に動作するクイズ番組の回答者です。
        1. 新しいコンテキストを使用して元の回答を**書き直す**。\n
        2. 新しいコンテキストが役に立たない場合は、元の回答を**繰り返す**。\n
        3. 質問に20文字以内で**答えだけ**を単語で簡潔に回答する。\n
        回答内で元の回答やコンテキストを直接参照しないでください。\n
        疑問がある場合は、元の答えを繰り返してください。
        <</SYS>>
        New Context: {context_msg}\n
        Query: {query_str}\n
        Original Answer: {existing_answer}\n
        New Answer: 
        [/INST]
        """)       

    if hybrid:
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            sparse_top_k=top_k_sparse,
            text_qa_template=qa_template,
            refine_template=refine_template,
            vector_store_query_mode="hybrid"
        )
        res_id = 'result_hybrid_top{}_{}'.format(top_k,top_k_sparse)
    else:
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            text_qa_template=qa_template,
            refine_template=refine_template,
        )
        res_id = 'result_top{}'.format(top_k)

    with open(questions,encoding='utf-8',mode='rt') as f:
        jsonl_data = [json.loads(l) for l in f.readlines()]

    with open(res_id+'.txt','wt') as fout:
        for i in tqdm(jsonl_data):
            res = query_engine.query(i['question'])
            src = res.get_formatted_sources(length=chunk_size)
            fout.write('-----{}-----'.format(i['qid']))
            fout.write(res.response+'\n')
            fout.write(src+'\n')
            fout.flush()
            i['prediction'] = res.response
            if 'answers' in i:
                del i['answers']

    with open(res_id+'.jsonl','wt') as fout:
        fout.write('\n'.join([json.dumps(l,ensure_ascii=False) for l in jsonl_data]))

service_context, storage_context = prepare_models()
index = make_embeddings(storage_context)
if index is None:
    print('## load index')
    if hybrid:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=storage_context_dir,storage_context=storage_context))
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=storage_context_dir))

generate_answers('datasets/aio_02_dev_v1.0.jsonl')

