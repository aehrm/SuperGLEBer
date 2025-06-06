#%%
import argparse
import collections
import itertools
from typing import List, Optional

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer, AutoTokenizer

import logging

from numpy.random import Generator

def group_by_context(dataset: Dataset) -> dict[str, List[dict]]:
    keyfun = lambda ex: (ex['context'], ex['question'])

    context_map = collections.defaultdict(list)
    for (context, question), examples in itertools.groupby(sorted(dataset.to_list(), key=keyfun), keyfun):
        examples = list(examples)
        ids = [ex['id'] for ex in examples]
        if len(examples) > 1:
            logging.warning(f"Examples {ids} have the same context and question! Skipping these.")
            continue

        to_add = examples[0]
        context_map[context].append(to_add)

    return context_map

def sample_niah(context_map: dict[str, list[dict]],
                needle: dict,
                needle_pos: int,
                haystack_size: int,
                random_gen: Generator = np.random.default_rng()) -> list[dict]:
    possible_haystack_contexts = list(sorted(set(context_map.keys()) - {needle['context']}))

    haystack_contexts = iter(random_gen.choice(possible_haystack_contexts, size=haystack_size-1, replace=False))

    haystack = []
    for i in range(haystack_size):
        if i == needle_pos:
            haystack.append(needle)
        else:
            sampled_context = next(haystack_contexts)
            sampled_distractor = random_gen.choice(context_map[sampled_context])
            haystack.append(sampled_distractor)

    return haystack

def format_niah(samples: list[dict], needle_pos: int) -> dict:
    haystack_context = []
    for sample in samples:
        haystack_context.append(sample['context'] + '\n\n###\n\n')

    needle_context_offset = sum(len(x) for x in haystack_context[:needle_pos])
    question = samples[needle_pos]['question']

    # only take the first answer!
    answers = {
        'answer_start': [samples[needle_pos]['answers']['answer_start'][0] + needle_context_offset],
        'text': [samples[needle_pos]['answers']['text'][0]]
    }

    return {
        'context': ''.join(haystack_context),
        'question': question,
        'answers': answers,
    }

def gen_niah_dataset(dataset: Dataset,
                     min_len: int = 2,
                     max_len: int = 15,
                     shuffle_source_dataset: bool = True,
                     random_gen: Generator = np.random.default_rng(),
                     max_sequence_length: Optional[int] = None,
                     tokenizer: Optional[PreTrainedTokenizer] = None):
    context_map = group_by_context(dataset)

    if shuffle_source_dataset:
        dataset = dataset.shuffle(generator=random_gen)

    for needle in dataset:
        size_ok = False
        while not size_ok:
            haystack_size = random_gen.integers(min_len, max_len+1)
            needle_pos = random_gen.integers(0, haystack_size)
            niah_sample = sample_niah(context_map, needle, needle_pos, haystack_size, random_gen)
            formatted = format_niah(niah_sample, needle_pos)
            if tokenizer is not None and max_sequence_length is not None:
                tokenized = tokenizer(formatted['question'], formatted['context'], truncation=False)
                if len(tokenized['input_ids']) < max_sequence_length:
                    size_ok = True
            else:
                size_ok = True

        yield formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('--max_sequence_length', type=int)
    parser.add_argument('--tokenizer', type=str)
    args = parser.parse_args()

    ds = load_dataset('deepset/germanquad')

    gen_kwargs = {}
    if args.max_sequence_length is not None:
        gen_kwargs['max_sequence_length'] = args.max_sequence_length
        gen_kwargs['tokenizer'] = AutoTokenizer.from_pretrained(args.tokenizer)

    out_dataset = DatasetDict({
        'train': Dataset.from_generator(gen_niah_dataset, gen_kwargs=dict(dataset=ds['train'], max_len=4)),
        'test': Dataset.from_generator(gen_niah_dataset, gen_kwargs=dict(**gen_kwargs, dataset=ds['test'], max_len=22))
    })

    out_dataset.save_to_disk(args.output_dir, max_shard_size="50MB")

if __name__ == '__main__':
    main()
