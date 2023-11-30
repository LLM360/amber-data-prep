import json
import multiprocessing as mp
from tqdm import tqdm

def worker_process(queue, result_list):
    """Process lines from the queue."""
    while True:
        try:
            line_id, line = queue.get()
            if line is None:  # Sentinel value indicating end of data
                break
            try:
                x = json.loads(line)
                if len(x['token_ids']) != 2049:
                    print(f"Bad length at {line_id}:", len(x['token_ids']))
                    result_list.append(line_id)
            except json.JSONDecodeError:
                print("Bad line:", line_id)
                result_list.append(line_id)
        finally:
            queue.task_done()

def validate_jsonl_file(filename):
    print("File:", filename)

    """Validate a jsonl file."""
    num_workers = mp.cpu_count()  # Adjust as needed
    queue = mp.JoinableQueue(maxsize=num_workers * 2)  # Buffer for lines
    manager = mp.Manager()
    result_list = manager.list()  # Shared list for results

    # Start worker processes
    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=worker_process, args=(queue, result_list))
        p.start()
        processes.append(p)

    # Feed lines to queue
    line_cnt = 0
    with open(filename, 'r') as f:
        for line_id, line in tqdm(enumerate(f, 1)):
            queue.put((line_id, line))
            line_cnt += 1

    # Signal end of data
    for _ in range(num_workers):
        queue.put((None, None))

    # Wait for all tasks to be processed
    queue.join()

    # Terminate processes
    for p in processes:
        p.terminate()

    print("Lines:", line_cnt)
    print(f"Failed lines: {list(result_list)}")
    print(f"====================")

if __name__ == "__main__":
    validate_jsonl_file('bad_sample.jsonl')
    validate_jsonl_file('good_sample.jsonl')

    # validate_jsonl_file('./redpajama_v1_llama_json_merge/arxiv/train.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/book/train.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/c4/train.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/github/train.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/refinedweb/train.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/stackexchange/train.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/starcoderdata/train.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/wikipedia/train.jsonl')

    # validate_jsonl_file('./redpajama_v1_llama_json_merge/arxiv/train_labeled.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/book/train_labeled.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/c4/train_labeled.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/github/train_labeled.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/refinedweb/train_labeled.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/stackexchange/train_labeled.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/starcoderdata/train_labeled.jsonl')
    # validate_jsonl_file('./redpajama_v1_llama_json_merge/wikipedia/train_labeled.jsonl')
