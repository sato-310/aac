import pickle
import re
import sys
from collections import defaultdict
from typing import List

import MeCab
from sklearn.naive_bayes import MultinomialNB


def extract_noun_from(sentence: str) -> List[str]:
    neologd = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

    lines = neologd.parse(sentence).split("\n")

    items = (re.split("[\t]", line) for line in lines)

    result = []
    for item in items:
        if len(item) != 2:
            continue

        if "名詞" in item[1].split(","):
            result.append(item[0])

    return result


if __name__ == "__main__":
    args = sys.argv
    filename = args[1]

    nouns = []
    with open(filename, "r") as f:
        for line in f:
            _, sentence = line.strip().split(",")

            nouns += extract_noun_from(sentence)

    X_train = []
    y_train = []
    with open(filename, "r") as f:
        for line in f:
            y, sentence = line.strip().split(",")
            X = []
            for noun in nouns:
                X.append(1 if noun in sentence else 0)
            X_train.append(X)
            y_train.append(int(y))

    model = MultinomialNB()
    model.fit(X_train, y_train)

    filename = "model.sav"
    pickle.dump(model, open(filename, 'wb'))

    filename = "model.sav"
    model = pickle.load(open(filename, 'rb'))

    # SAMPLE
    sentence = "安倍は辞めろ。" # アベガー
    sentence = "安倍総理がんばれ！" # 安倍信者

    data = []
    for noun in nouns:
        data.append(1 if noun in sentence else 0)

    response = model.predict([data])

    if response:
        print("アベガー")
    else:
        print("安倍信者")
