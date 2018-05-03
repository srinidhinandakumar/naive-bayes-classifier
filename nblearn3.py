import json
import time
import string
import pprint
import sys
import re


class NaiveBayesLearn:

    def __init__(self):
        self.total_label_counts = dict()
        self.total_label_counts["True"] = 0
        self.total_label_counts["Fake"] = 0
        self.total_label_counts["Pos"] = 0
        self.total_label_counts["Neg"] = 0
        self.word_label_counts = dict()
        self.label1_counts = dict()
        self.label2_counts = dict()
        self.label1_counts["True"] = 0
        self.label1_counts["Fake"] = 0
        self.label2_counts["Pos"] = 0
        self.label2_counts["Neg"] = 0
        self.stopwords = []

    def read_data(self, filename):
        fp = open(filename)
        data = fp.read()
        return data

    def process_data(self, data):
        lines = data.split("\n")
        train_data = dict()
        for l in lines:
            if l == "":
                continue
            else:
                id_and_text = l.split(" ", 1)
                label1_and_text = id_and_text[1].split(" ", 1)
                label2_and_text = label1_and_text[1].split(" ", 1)
                if id_and_text[0] not in train_data:
                    train_data[id_and_text[0]] = dict()
                    # replace punctuation with space
                    # label2_and_text[1] = re.sub("[^\w\d'\s]+", '', label2_and_text[1])
                    label2_and_text[1] = label2_and_text[1].translate(str.maketrans(" ", " ", string.punctuation))
                    # replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
                    # label2_and_text[1] = label2_and_text[1].translate(replace_punctuation)
                    # # replace numbers with space
                    label2_and_text[1] = label2_and_text[1].translate(str.maketrans(' ', ' ', string.digits))
                    # replace_digits = str.maketrans(string.digits, ' ' * len(string.digits))
                    # label2_and_text[1] = label2_and_text[1].translate(replace_digits)
                    #
                    train_data[id_and_text[0]][str("Review")] = label2_and_text[1].lower()
                    train_data[id_and_text[0]][str("Label 1")] = label1_and_text[0]
                    train_data[id_and_text[0]][str("Label 2")] = label2_and_text[0]
                else:
                    continue

        return train_data

    def learn(self, text, label1, label2):
        words = text.split(" ")
        for w in words:
            if w in self.word_label_counts:
                # label1
                if label1 in self.word_label_counts[w]:
                    self.word_label_counts[w][label1] += 1
                else:
                    self.word_label_counts[w][label1] = 1
                # label2
                if label2 in self.word_label_counts[w]:
                    self.word_label_counts[w][label2] += 1
                else:
                    self.word_label_counts[w][label2] = 1
            else:
                self.word_label_counts[w] = dict()
                self.word_label_counts[w][label1] = 1
                self.word_label_counts[w][label2] = 1

        # label counts
        self.total_label_counts[label1] += len(words)
        self.total_label_counts[label2] += len(words)
        self.label1_counts[label1] += 1
        self.label2_counts[label2] += 1

    def fit(self, train_data):
        for each_id in train_data:
            self.learn(train_data[each_id]["Review"], train_data[each_id]["Label 1"], train_data[each_id]["Label 2"])
        labels = ["True", "Fake", "Pos", "Neg"]
        for word in self.word_label_counts:
            for label in labels:
                if label not in self.word_label_counts[word]:
                    self.word_label_counts[word][label] = 0

    def remove_stopwords(self):
        stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"]
        self.stopwords = stopwords


if __name__ == "__main__":
    t1 = time.time()
    model = NaiveBayesLearn()
    # filename = sys.argv[1]
    filename = "../coding-2-data-corpus/train-labeled.txt"
    outputfile1 = "nbmodel.json"
    outputfile2 = "nboutput.txt"
    data = model.read_data(filename)
    train = model.process_data(data)
    model.fit(train)
    model.remove_stopwords()
    result = dict()
    result["label1-counts"] = model.label1_counts
    result["label2-counts"] = model.label2_counts
    result["word-label-counts"] = model.word_label_counts
    result["total-label-counts"] = model.total_label_counts
    result["stopwords"] = model.stopwords
    with open(outputfile1, 'w') as outfile:
        json.dump(result, outfile, indent=4)
    print("time: ", (time.time() - t1))