# This script will 
# 1. tokenize the text of a subset
# 2. concatenate the tokens to a given length (2048)
# 3. write the token ids to a jsonl file
# 4. repeat for all subsets

import os
import platform
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Union

import datasets as hf_datasets
import psutil
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from functools import partial
import os
import warnings
from typing import Dict, Iterable, Union

import datasets as hf_datasets
import numpy as np
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase
import json

class ConcatTokensDataset(IterableDataset):

    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('',
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                'in duplicated special tokens. Please be sure this is what you intend.'
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        buffer = []
        for sample in self.hf_dataset:
            encoded = self.tokenizer(sample['text'],
                                     truncation=False,
                                     padding=False, add_special_tokens=False)
            iids = encoded['input_ids']
            buffer = buffer + self.bos_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:] if self.should_wrap else []
                yield {
                    'tokens': np.asarray(concat_sample).tobytes()
                }

# redpajama_v1
# arxiv  book  c4  github  stackexchange   wikipedia
subfolders = [
    'redpajama_v1/arxiv',
    'redpajama_v1/book',
    'redpajama_v1/c4',
    'redpajama_v1/github',
    'redpajama_v1/stackexchange',
    'redpajama_v1/wikipedia',

    # 2019-30  2020-05  2021-04  2022-05  2023-06
    # 'redpajama_v2/common_crawl/2019-30',
    # 'redpajama_v2/common_crawl/2020-05',
    # 'redpajama_v2/common_crawl/2021-04',
    # 'redpajama_v2/common_crawl/2022-05',
    # 'redpajama_v2/common_crawl/2023-06',
]

# falcon-refinedweb
subfolders += [
    'falcon-refinedweb/data'
]

# starcoder
with open('starcoder-lang.list') as flist:
    subfolders += [f"starcoderdata/{l.strip()}" for l in flist.readlines()]


print(subfolders)

class ConcatMode(Enum):
    NO_CONCAT = "NO_CONCAT"
    CONCAT_TOKENS = "CONCAT_TOKENS"


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description="Convert dataset into MDS format, optionally concatenating and tokenizing"
    )
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--concat_tokens",
        type=int,
        help="Convert text to tokens and concatenate up to this many tokens",
    )

    parser.add_argument("--tokenizer", type=str, required=False, default=None)
    parser.add_argument("--bos_text", type=str, required=False, default=None)
    parser.add_argument("--eos_text", type=str, required=False, default=None)
    parser.add_argument("--no_wrap", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, required=False, default=None)

    parsed = parser.parse_args()

    # Make sure we have needed concat options
    if (
        parsed.concat_tokens is not None
        and isinstance(parsed.concat_tokens, int)
        and parsed.tokenizer is None
    ):
        parser.error("When setting --concat_tokens, you must specify a --tokenizer")

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ""
    if parsed.eos_text is None:
        parsed.eos_text = ""
    return parsed


def build_hf_dataset(
    dataset_name: str,
    split: str,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = "",
    eos_text: str = "",
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase = None,
    sub: str = None,
) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """
    hf_dataset = hf_datasets.load_dataset(
        path=dataset_name, data_dir=sub, split=split, streaming=True
    )
    # num_rows = len(hf_dataset)
    # print('Number of samples: ', num_rows)
    if 'refinedweb' in sub or 'starcoder' in sub:
        hf_dataset = hf_dataset.rename_column("content", "text")

    if mode == ConcatMode.NO_CONCAT:
        raise
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(f"{tokenizer=} must be of type PreTrainedTokenizerBase")
        if max_length is None:
            raise ValueError(f"max_length must be set.")
        if bos_text + eos_text == "":
            test_tokens = tokenizer("test")
            if (
                test_tokens["input_ids"][0] != tokenizer.bos_token_id
                and test_tokens["input_ids"][-1] != tokenizer.eos_token_id
            ):
                tok_error_msg = "This tokenizer does not insert an EOS nor BOS token. "
                tok_error_msg += (
                    "Concatenating with this tokenizer will result in sequences being "
                )
                tok_error_msg += "attached without a separating token. Please use another tokenizer, "
                tok_error_msg += (
                    "such as facebook/opt-125m, or specify EOS/BOS text with e.g. "
                )
                tok_error_msg += "--bos_text=<|endoftext|>."
                raise ValueError(tok_error_msg)
        dataset = ConcatTokensDataset(
            hf_dataset=hf_dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
        )
    return dataset


def build_dataloader(dataset, batch_size, num_workers) -> DataLoader:
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if "linux" or "macos" in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())  # type: ignore
        else:
            num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size // num_workers) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(
    loader: DataLoader, truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


import multiprocessing


def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """

    for sub in subfolders:
        process_sub(sub, args)

    # with multiprocessing.Pool(1) as pool:
    #     pool.map(partial(process_sub, args=args), all_languages)


def process_sub(sub, args):
    print(f"Processing {sub}...")

    if os.path.exists(os.path.join(args.out_root, sub)):
        print(f"Skipping {sub} because it already exists...")
        return
    else:
        os.makedirs(os.path.join(args.out_root, sub), exist_ok=False)


    if args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # we will enforce length, so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
        columns = {"tokens": "bytes"}
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None
        columns = {"text": "str"}

    hf_split = "train"

    # Get samples
    dataset = build_hf_dataset(
        dataset_name=args.input_root,
        split=hf_split,
        mode=mode,
        max_length=args.concat_tokens,
        bos_text=args.bos_text,
        eos_text=args.eos_text,
        no_wrap=args.no_wrap,
        tokenizer=tokenizer,
        sub=sub,
    )
    loader = build_dataloader(
        dataset=dataset, batch_size=512, num_workers=args.num_workers
    )
    samples = generate_samples(loader)

    # Write samples
    print(f"Converting {sub} to jsonl ...")
    out_filename = os.path.join(args.out_root, sub, 'train.jsonl')
    print('Output:', out_filename)
    with open(out_filename+'.tmp', 'w') as fout:
        for sample in tqdm(
            samples,
            desc=sub,
            #    total=num_rows
        ):
            json_sample = { "token_ids": np.frombuffer(sample['tokens'], dtype=np.int64).tolist(), 'source': sub}
            out_line = json.dumps(json_sample) + '\n'
            fout.write(out_line)
        
    os.rename(out_filename+'.tmp', out_filename)
    print(f"Converting {sub} to jsonl format completed")


if __name__ == "__main__":
    main(parse_args())
