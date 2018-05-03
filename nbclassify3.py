import json
import time
import string
import pprint
import math
import sys
import re


class NaiveBayesPredict:
    def __init__(self):
        self.probabilities = dict()
        self.counts = dict()
        self.results = dict()
        self.n = 0

    def read_data(self, filename):
        fp = open(filename)
        data = fp.read()
        return data

    def process_data(self, data):
        lines = data.split("\n")
        test_data = dict()
        for line in lines:
            if line == "":
                continue
            else:
                id_and_text = line.split(" ", 1)
                if id_and_text[0] not in self.results:
                    self.results[id_and_text[0]] = dict()
                    self.results[id_and_text[0]]["label1"] = ""
                    self.results[id_and_text[0]]["label2"] = ""
                    text = id_and_text[1].lower()
                    # replace punctuation with space
                    # text = re.sub("[^\w\d'\s]+", '', text)
                    text = text.translate(str.maketrans(" ", " ", string.punctuation))
                    # replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
                    # text = text.translate(replace_punctuation)

                    # replace numbers with space
                    text = text.translate(str.maketrans(' ', ' ', string.digits))
                    # replace_digits = str.maketrans(string.digits, ' ' * len(string.digits))
                    # text = text.translate(replace_digits)

                    test_data[id_and_text[0]] = text
                else:
                    continue
        return test_data

    def evaluate(self, text):
        words = text.split(" ")
        label1 = ["True", "Fake"]
        label2 = ["Pos", "Neg"]
        choice1 = dict()
        # Label 1
        for label in label1:
            prob = 0
            for w in words:
                if w in self.counts["stopwords"]:
                    # print("a\t")
                    continue
                else:
                    if w in self.counts["word-label-counts"]:
                        prob += math.log((self.counts["word-label-counts"][w][label] + 1) / (self.counts["total-label-counts"][label] + self.n))
                    else:
                        prob += 0
            prob += math.log(self.counts["label1-counts"][label])
            choice1[label] = prob
        l1 = max(choice1.items(), key=lambda x: x[1])[0]  # choose max label
        # Label 2
        choice2 = dict()
        for label in label2:
            prob = 0
            for w in words:
                if w in self.counts["stopwords"]:
                    # print("a\t")
                    continue
                else:
                    if w in self.counts["word-label-counts"]:
                        prob += math.log((self.counts["word-label-counts"][w][label] + 1) / (self.counts["total-label-counts"][label] + self.n))
                    else:
                        prob += 0
            prob += math.log(self.counts["label2-counts"][label])
            choice2[label] = prob
        l2 = max(choice2.items(), key=lambda x: x[1])[0]  # choose max label

        return l1, l2

    def classify(self, test_data):
        # print(self.counts["stopwords"])
        for line_id in test_data:
            label1, label2 = self.evaluate(test_data[line_id])
            self.results[line_id]["label1"] = label1
            self.results[line_id]["label2"] = label2


if __name__ == "__main__":
    t1 = time.time()
    filename1 = "nbmodel.json"
    filename2 = "nboutput.txt"
#     filename3 = "../coding-2-data-corpus/dev-text.txt"
    filename3 = sys.argv[1]
    counts = json.load(open(filename1))
    model = NaiveBayesPredict()
    model.counts = counts
    model.n = len(counts["word-label-counts"].keys())
    data = model.read_data(filename3)
    test_data = model.process_data(data)
    model.classify(test_data)
    st = ""
    for line_id in model.results:
        st += line_id + " " + model.results[line_id]["label1"] + " " + model.results[line_id]["label2"] + "\n"
    # print(st)
    fp = open(filename2, "w")
    fp.write(st)
    print("time: ", str(time.time() - t1))

