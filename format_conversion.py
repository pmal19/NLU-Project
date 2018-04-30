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


import csv
filename=open("/Users/anhadmohananey/Downloads/questions.csv", "r")
reader=csv.DictReader(filename, delimiter=",")
arr=[]
def parse(x):
	return x.lower().replace(".", " .").replace("?", " ?").replace("!", " !")
with open("all.csv", "w") as f:
	for x in reader:
		print("%s\t%s\t%s" % (parse(x["question1"]), parse(x["question2"]), x["is_duplicate"]), file=f)