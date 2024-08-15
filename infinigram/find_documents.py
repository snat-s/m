from datasets import load_dataset
import requests

benchmarks = [
    # common sense
    # "cais/mmlu",
    "Rowan/hellaswag",
    "winogrande",
    "allenai/ai2_arc",
    # knowledge
    "truthful_qa",
    # mandarjoshi/trivia_qa, #  pretty BIG
    "openai_humaneval",
]


def query_api(example):
    query = example['ctx']
    payload = {"corpus": "v4_dolma-v1_6_llama", "query_type": "count", "query": query}
    result = requests.post("https://api.infini-gram.io/", json=payload).json()
    # print(result)
    example['api_count'] = result['count']
    #print(result['count'])
    return example

for benchmark in benchmarks:
    if "winogrande" in benchmark:
        dataset = load_dataset(benchmark, "winogrande_xl")
    elif "arc" in benchmark:
        dataset = load_dataset(benchmark, "ARC-Challenge")
    elif "qa" in benchmark:
        dataset = load_dataset(benchmark, "generation")
    else:
        dataset = load_dataset(benchmark)
    
    dataset["test"] = dataset["test"].map(
        query_api, num_proc=32
    )
    benchmark_name = benchmark.split("/")[-1]
    dataset.save_to_disk(f"{benchmark_name}_processed")

    filtered_examples = [
        example for example in dataset["test"] if example["api_count"] > 1
    ]

    for example in filtered_examples:
        print(example)
        print()

    break
