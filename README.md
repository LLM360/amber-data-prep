<!-- # usage

1. Use `convert_dataset_hf_refinedpajama_json.py` to tokenize raw text documents and concat tokens to 2048 tokens. Use the `subfolders` in code to specify the subsets.
2. Use `mix_and_split.py` to merge subsets, then split the merged dataset into 360 chunks
3. Use `validate_json.py` to validate data samples. -->


# Overview

This repository contains the code to prepare the data for the **Amber 7B language model**. 

The final training data
comes from three sources:
1. RedPajama V1 (we use the `arxiv`, `book`, `c4`, `github`, `stackexchange`, and `wikipedia` subsets)
2. RefinedWeb (we use this to replace the common_crawl subset of RedPajama V1)
3. StarCoderData

The data is prepared in the following steps:
1. Download the untokenized data from the sources.
2. Tokenize the data using the Huggingface tokenizer (`LLaMA` tokenizer in our case).
3. Concatenate the tokens into 2048 token sequences.
4. Merge the datasets together and split the merged dataset into `360` chunks.

A tokenized data chunk will be a jsonl file like this:

```json
{"token_ids": [660, 29901, ...(more token ids)..., 29901], "source": "c4"} # first sample in chunk
{"token_ids": [29896, 29946, ...(more token ids)..., 13], "source": "arxiv"} # second sample in chunk
```

Each sample will have `2049` tokens (to make it easier to get the shifted labels next token prediction). The `source` field indicates which dataset the sample comes from, which is optional.

Check out the `good_sample_label.jsonl` to see a sample of the final data.

# Download untokenized data

This section describes how to download the untokenized data from the sources.

## 1. Download RedPajama V1:

We have provided a script to download the RedPajama V1 dataset. To download the dataset, run the following commands:

```bash
cd redpajama_v1
./download.sh
```
This will download the whole dataset into the `redpajama_v1` folder using `wget`. We don't need the "common_crawl" subset of the dataset since we replace it with the RefinedWeb dataset.

To avoid downloading the whole dataset, you can manually modify the `urls.txt` file to remove the lines with the "common_crawl".

(The `urls.txt` is obtained by running: `wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'`)


## 2. Download RefinedWeb

The RefinedWeb dataset we use is https://huggingface.co/datasets/tiiuae/falcon-refinedweb.
You can use whatever way you want to download the dataset from huggingface reporsitory. For example, the most straightforward way is to use:

```bash
# make sure you have "git-lfs" installed.
git clone https://huggingface.co/datasets/tiiuae/falcon-refinedweb
```


## 3. Download StarCoderData:
We also use StarCoderData https://huggingface.co/datasets/bigcode/starcoderdata in our training.

To download:
```bash
git clone https://huggingface.co/datasets/bigcode/starcoderdata
```

# Tokenize and Concatenate Sequences

You can use the script `convert_dataset_hf_refinedpajama_json.py` to tokenize and concatenate the sequences. The script will tokenize the raw text documents and concatenate the tokens into 2048 token sequences. Each subset will be processed separately and the output will be saved in a separate folder. 

```bash
python convert_dataset_hf_refinedpajama_json.py \
  --input_root ./ \
  --out_root refinedpajama_llama_json \
  --concat_tokens 2049 \
  --tokenizer huggyllama/llama-7b \
  --num_workers 64 \
  --eos_text '</s>' 
```

**Known issue**: Some documents can be too long, and the tokenizer is very slow when processing long sequences. The progress of `convert_dataset_hf_refinedpajama_json.py` will appear to be stuck at some point. That's because the tokenizer of some workers are still processing the long sequences and other workers are waiting for them to finish. You can wait for the workers to finish or mannually break the document into smaller chunks before tokenization and process them separately.

After the script finishes, each subset will be saved in a separate folder with `train.jsonl` (for example, `<out_root>/redpajama_v1/arxiv/train.jsonl`).


# Validate the Data (Optional)

You can use the script `validate_json.py` to validate the data samples. The script will check if the samples are valid json and if the samples have the correct number of tokens. 

```bash
python validate_json.py
```

You can modify the path in the script's main function to specify the jsonl file to validate.

# Merge Subsets and Split into Chunks

After the subsets are tokenized and concatenated, we can merge them together and split the merged dataset into chunks. We use the script `mix_and_split.py` to merge subsets and split the merged dataset into chunks. The script will merge the subsets together and split the merged dataset into `360` chunks. The output will be saved in a separate folder.

```bash
python mix_and_split.py \
  --input_root refinedpajama_llama_json \
  --out_root redpajama_v1_llama_json_merged_360 \
  --subfolders='refinedpajama_llama_json/redpajama_v1/arxiv,refinedpajama_llama_json/redpajama_v1/c4,...' \ # specify the subsets to merge, comma separated
  --num_chunks 360 \
  --num_valid_samples_per_subfolder 102400 # 100K samples
```

The logic of the script is to iterate each tokenized subset and distribute the samples to the chunks. For example, the first sample of the `arxiv` will goto 1st chunk, the second sample of the `arxiv` will goto 2nd chunk, and so on. 

Each chunk will end up with samples from all the subsets in the order of the subsets specified in `--subfolders`. For example, the first part of a chunk will be from the first subset, the second part of that chunk will be from the second subset, and so on. This means shuffling within a chunk is needed. **The script itself doesn't do shuffling within a chunk**. You can add shuffling in the trainig code or perform shuffling before training.

