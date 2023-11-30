"""
This script will merge jsonl files from different subsets (e.g., arxiv, c4, wikipedia, etc.) into one output directory (output_root).
The output directory will have two subfolders: train and valid. 
The train folder will contain 360 files, each containing 1/360 of the training data.

Shuffle is NOT done by this script

"""

import json
import os
from tqdm import tqdm
import random
import fire

def main(input_root, output_root, subfolders, num_split, num_valid_samples_per_subfolder):
    os.makedirs(output_root, exist_ok=False)
    train_dir = os.path.join(output_root, 'train')
    valid_dir = os.path.join(output_root, 'valid')
    os.makedirs(train_dir, exist_ok=False)
    os.makedirs(valid_dir, exist_ok=False)

    subfolders=subfolders.split(',')
    for subfolder in subfolders:
        subfolder_input_file = os.path.join(input_root, subfolder,'train_labeled.jsonl')

        # Calculate total lines in the file
        print(f"Counting lines in: {subfolder}")
    
        total_lines = 0
        with open(subfolder_input_file) as fin:
            for _ in tqdm(fin):
                total_lines += 1

        num_valid_samples = num_valid_samples_per_subfolder
        num_train_samples = total_lines - num_valid_samples

        num_train_per_split = num_train_samples//num_split
        num_unused_samples = total_lines - num_train_per_split * num_split - num_valid_samples

        num_used_samples = total_lines - num_unused_samples

        # random sample num_valid_samples index from range(used_samples)
        valid_sample_idx = set(random.sample(range(num_used_samples), num_valid_samples))
        assert len(valid_sample_idx) == num_valid_samples

        
        # output_split_idx = 0
        num_written_train = 0
        num_written_valid = 0

        valid_filename = os.path.join(valid_dir, subfolder+'.jsonl')

        print(f'Num samples per train split: {num_train_per_split}')
        print(f'Will write {num_valid_samples} valid samples to: {valid_filename}')

        print(f"Creating {num_split} output files, ranging from {0} to {num_split-1}")

        output_file_pool = [ open(os.path.join(train_dir, f'train_{output_split_idx}.jsonl'), 'a') for output_split_idx in range(num_split)]

        out_valid_file = open(valid_filename,'w')
        out_train_file = None
        with open(subfolder_input_file) as fin:
            progress = tqdm(enumerate(fin), total=num_used_samples)
            for line_no, line in progress:
                if line_no >= num_used_samples:
                    break

                # add source
                # line = json.loads(line)
                # line['source'] = subfolder
                # line = json.dumps(line)+'\n'

                if line_no in valid_sample_idx:
                    # a valid sample
                    out_valid_file.write(line)
                    num_written_valid+=1
                else:
                    # a train sample
                    # if out_train_file is None:
                    #     train_filename = os.path.join(train_dir, f'train_{output_split_idx}.jsonl')
                    #     progress.set_description(f'Append to {train_filename}')
                    #     out_train_file = open(train_filename, 'a')
                    output_split_idx = num_written_train % num_split
                    out_train_file = output_file_pool[output_split_idx]
                    out_train_file.write(line)
                    num_written_train+=1

                    # enough for this split
                    # if num_written_train % num_train_per_split == 0:
                    #     out_train_file.close()
                    #     output_split_idx += 1
                    #     out_train_file = None

        # when all is completed
        # out_valid_file.close()

        print("Closing all files ...")

        for out_valid_file in output_file_pool:
            out_valid_file.close()
        
        print("All files are closed")

        # assert out_train_file is None # should be closed
        # assert output_split_idx == num_split - 1    
        assert num_written_valid + num_written_train == num_used_samples

        print("Num train samples:", num_written_train)
        valid_percent = num_written_valid/ (num_written_valid+num_written_train) * 100
        print("Num valid samples:", num_written_valid, f"({valid_percent:.3f}%)")
        print(f"Completed {subfolder}")
        print("===================")
                    
                
if __name__ == "__main__":
    # main(
    #     input_root = './redpajama_v1_llama_json_merge', # CHANGE THIS
    #      output_root = './redpajama_v1_llama_json_360s_shuffle', # CHANGE THIS
    #      subfolders='arxiv,book,c4,refinedweb,stackexchange,starcoderdata,wikipedia',
    #      num_split=360,
    #      num_valid_samples_per_subfolder=100*1024   # 100K samples / 200M tokens
    #     )
    fire.Fire(main)
