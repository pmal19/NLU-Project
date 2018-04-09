import numpy as np
import json
import codecs

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    # Used in the unlabeled test set---needs to map to some arbitrary label.
    "hidden": 0,
}


def load_nli_data(path, choose=lambda x: True):
    print "Loading", path
    examples = []
    failed_parse = 0
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue

            if not choose(loaded_example):
                continue

            example = {}
            example["label"] = loaded_example["gold_label"]
            example["sentence_1"] = loaded_example["sentence1"]
            example["sentence_2"] = loaded_example["sentence2"]
            examples.append(example)
    return examples
    
def load_sst_data(
        path):
    dataset = convert_unary_binary_bracketed_data(path)
    return dataset

def convert_unary_binary_bracketed_data(
        filename,
        keep_fn=lambda x: True,
        convert_fn=lambda x: x):
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            example = {}
            line = line.strip()
            if len(line) == 0:
                continue
            label = line[1]
            if not keep_fn(label):
                continue
	        label = convert_fn(label)
            example["label"] = label
            example["sentence"] = line
            example["tokens"] = []
            example["transitions"] = []
            words = example["sentence"].replace(')', ' )')
            words = words.split(' ')

            for index, word in enumerate(words):
                if word[0] != "(":
                    if word == ")":
                        # Ignore unary merges
                        if words[index - 1] == ")":
                            example["transitions"].append(1)
                    else:
                        # Downcase all words to match GloVe.
                        example["tokens"].append(word.lower())
                        example["transitions"].append(0)
            example["example_id"] = str(len(examples))
            example["sentence_1"]=" ".join(example["tokens"])
            for k in ['example_id','sentence','tokens', 'transitions']:
            	del(example[k])
            examples.append(example)
    return examples