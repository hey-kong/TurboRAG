import logging
import sys
import torch
import json
import time
import asyncio
from tabulate import tabulate
import argparse
from transformers import AutoTokenizer
from qwen2 import Qwen2ModifiedForCausalLM

# Llama Index Related
from llama_index.core import Settings, load_index_from_storage, StorageContext, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def qa_to_prompt(chunk_text, query):
    prompt = f'''{chunk_text}

    Analyze the above document and determine whether the document is relevant for answering the question: {query}

    If the document provides information relevant to answering the question, generate ‘True’.
    If there is no relevant information, generate ‘False’ instead.

    Answer:
    '''
    return prompt


# Parse command-line arguments at global scope
parser = argparse.ArgumentParser(description='RAG with KV Cache Benchmarking Script')
parser.add_argument('--model_name', type=str, help='Path to the pretrained language model')
parser.add_argument('--embedding_model_name', type=str, help='Embedding model name or path')
parser.add_argument('--storage_dir', type=str, default='doc_emb', help='Directory where the index storage is located')
parser.add_argument('--query_file', type=str, default='./questions/query.jsonl',
                    help='Path to the file containing queries')
parser.add_argument('--num_questions', type=int, default=50, help='Number of questions to process')
parser.add_argument('--similarity_top_k', type=int, default=20, help='Number of topk most relevant chunks')
parser.add_argument('--use_flash_attn', action='store_true', help='Use FlashAttention2')
parser.set_defaults(use_chunk_cache=True)
args = parser.parse_args()

# Set up device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer globally
attn_implementation = "flash_attention_2" if args.use_flash_attn else None
model = Qwen2ModifiedForCausalLM.from_pretrained(
    args.model_name,
    attn_implementation=attn_implementation,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model.config.use_cache = True

# Set up embedding model and index
Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_name)
storage_context = StorageContext.from_defaults(persist_dir=args.storage_dir)
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=args.similarity_top_k)


def load_kvcache(cache_file_path):
    return torch.load(cache_file_path, weights_only=True)


async def async_load_kvcache(cache_file_path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, load_kvcache, cache_file_path)


async def query_with_kvcache(query_text, use_chunk_cache=True):
    query_bundle = QueryBundle(query_str=query_text)
    retrieved_nodes = retriever.retrieve(query_bundle)

    kvcache_futures = {}

    left = 0
    right = len(retrieved_nodes) - 1
    last_true_index = -1
    while left <= right:
        mid = (left + right) // 2
        node_with_score = retrieved_nodes[mid]
        node = node_with_score.node

        kvcache = None
        if use_chunk_cache:
            if mid not in kvcache_futures:
                kvcache = load_kvcache(node.metadata["kvcache_file_path"])
            else:
                kvcache = await kvcache_futures.pop(mid)
            mid_future = (left + mid - 1) // 2
            if mid_future >= left and mid_future not in kvcache_futures:
                kvcache_futures[mid_future] = asyncio.create_task(async_load_kvcache(retrieved_nodes[mid_future].node.metadata["kvcache_file_path"]))
            mid_future = (mid + 1 + right) // 2
            if mid_future <= right and mid_future not in kvcache_futures:
                kvcache_futures[mid_future] = asyncio.create_task(async_load_kvcache(retrieved_nodes[mid_future].node.metadata["kvcache_file_path"]))

        prompt = qa_to_prompt(node.text, query_text)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        eos_token_ids = [151645, 151643]
        with torch.no_grad():
            start = time.perf_counter()
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                past_key_values=kvcache,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                eos_token_id=eos_token_ids,
                use_cache=use_chunk_cache,
            )
            end = time.perf_counter()
            use_time_without_cache = end - start
            print(f"{use_time_without_cache:.6f} seconds")
            generated_ids = outputs[0]
            new_token_id = generated_ids[-1].item()
            new_token = tokenizer.decode(new_token_id).strip()
            print(new_token)

        if new_token == "True":
            last_true_index = mid
            left = mid + 1
        else:
            right = mid - 1

    if use_chunk_cache:
        # clear other tasks
        for task in kvcache_futures.values():
            if not task.done():
                task.cancel()
        # wait for all tasks to complete, to avoid resource leak
        try:
            await asyncio.gather(*kvcache_futures.values(), return_exceptions=True)
        except Exception:
            pass  # ignore exceptions
        kvcache_futures.clear()
    print(last_true_index)


async def main():
    questions = []
    with open(args.query_file) as file:
        for item in file:
            data = json.loads(item)
            questions.append(data["query"])
    questions = questions[:args.num_questions]

    # Test the average time taken for RAG without document chunk KV Cache
    start = time.perf_counter()
    for query in questions:
        await query_with_kvcache(query, use_chunk_cache=False)
    end = time.perf_counter()
    use_time_without_cache = end - start
    avg_time_without_cache = use_time_without_cache / len(questions)

    # Test the average time taken for RAG with document chunk KV Cache
    start = time.perf_counter()
    for query in questions:
        await query_with_kvcache(query)
    end = time.perf_counter()
    use_time = end - start
    avg_time_with_cache = use_time / len(questions)

    # Prepare data for tabular output
    results = [
        ["Without KV Cache", f"{avg_time_without_cache:.6f} seconds"],
        ["With KV Cache", f"{avg_time_with_cache:.6f} seconds"]
    ]

    # Print the results in a table format
    print(tabulate(results, headers=["Method", "Average Time"], tablefmt="grid"))


asyncio.run(main())