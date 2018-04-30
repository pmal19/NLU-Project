import dataparser as dp
from __future__ import print_function
data=dp.load_nli_data("/Users/anhadmohananey/Downloads/snli_1.0/snli_1.0_test.jsonl")
LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    # Used in the unlabeled test set---needs to map to some arbitrary label.
    "hidden": 0,
}

with open("test.tsv", "w") as f:
	for x in data:
		print ("%s\t%s\t%s" % (x['sentence_1'], x["sentence_2"], LABEL_MAP[x["label"]]), file=f)