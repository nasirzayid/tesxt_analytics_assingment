from collections import Counter
from fractions import Fraction
from pathlib import Path
import math
import matplotlib.pyplot as plt


def heading(title):
    print("\n" + title)
    print("-" * len(title))


# load the data
here = Path(__file__).resolve().parent
found = list(here.rglob("Data_3.txt")) or list(here.parent.rglob("Data_3.txt"))
if not found:
    raise FileNotFoundError(f"Could not find Data_3.txt - put it in {here}")

with open(found[0], encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip().startswith("<s>")]


# pre processing lowercase and split on spaces keeping <s> and </s>
def tokenize(sentence):
    return sentence.lower().split()

train = [tokenize(line) for line in lines[:-1]]
test = tokenize(lines[-1])

heading("Training sentences")
for i, sent in enumerate(train, start=1):
    print(f"S{i}: {' '.join(sent)}")

heading("Test sentence")
print(" ".join(test))


# count single words and word pairs
word_count = Counter()
pair_count = Counter()

for sent in train:
    word_count.update(sent)
    pair_count.update(zip(sent, sent[1:]))   # pairs stay inside one sentence

V = len(word_count)

heading("Word counts")
for word, n in word_count.items():
    print(f"{word:<10} {n}")
print(f"\nVocabulary size V = {V}")

heading("Bigram counts")
for (w1, w2), n in pair_count.items():
    print(f"{w1 + ' ' + w2:<16} {n}")


# the two models
def mle(prev, word):
    if word_count[prev] == 0:
        return Fraction(0)
    return Fraction(pair_count[(prev, word)], word_count[prev])

def laplace(prev, word):
    return Fraction(pair_count[(prev, word)] + 1, word_count[prev] + V)

def sentence_prob(sent, model):
    prob = Fraction(1)
    for prev, word in zip(sent, sent[1:]):
        prob *= model(prev, word)
    return prob

def show(prob):
    return f"{prob} = {float(prob):.10f}"


# to test sentence, one bigram at a time
heading("Test sentence, bigram by bigram")
print(f"{'Bigram':<22}{'Unsmoothed':<12}Add-1")
for prev, word in zip(test, test[1:]):
    print(f"{f'P({word} | {prev})':<22}{str(mle(prev, word)):<12}{laplace(prev, word)}")

p_mle = sentence_prob(test, mle)
p_add1 = sentence_prob(test, laplace)

print(f"\nUnsmoothed : {show(p_mle)}")
print(f"Add-1      : {show(p_add1)}")
print(f"log P (unsmoothed) = {math.log(p_mle):.4f}")
print(f"log P (add-1)      = {math.log(p_add1):.4f}")


# to compare every sentence
heading("All sentences")
labels = ["S1", "S2", "S3", "Test"]
mle_values, add1_values = [], []

for label, sent in zip(labels, train + [test]):
    p1 = sentence_prob(sent, mle)
    p2 = sentence_prob(sent, laplace)
    mle_values.append(float(p1))
    add1_values.append(float(p2))
    print(f"{label:<5} {' '.join(sent):<46} {str(p1):<6} {float(p2):.4e}")


# a sentence with word pairs the model has never seen
heading("Why smoothing matters")
unseen = tokenize("<s> Danielle read a book </s>")
print("Sentence:", " ".join(unseen), "\n")

for prev, word in zip(unseen, unseen[1:]):
    note = "  <- never seen" if pair_count[(prev, word)] == 0 else ""
    print(f"{f'P({word} | {prev})':<22}{str(mle(prev, word)):<6}{str(laplace(prev, word)):<6}{note}")

print(f"\nUnsmoothed : {show(sentence_prob(unseen, mle))}")
print(f"Add-1      : {show(sentence_prob(unseen, laplace))}")


#bar chart log scale because the smoothed values are tiny
x = range(len(labels))
plt.figure(figsize=(8, 4.5))
plt.bar([i - 0.2 for i in x], mle_values, 0.4, label="Unsmoothed")
plt.bar([i + 0.2 for i in x], add1_values, 0.4, label="Add-1 smoothed")
plt.yscale("log")
plt.xticks(list(x), labels)
plt.ylabel("Probability (log scale)")
plt.title("Sentence probabilities: unsmoothed vs add-1")
plt.legend()
plt.tight_layout()
plt.savefig(here / "q4_chart.png", dpi=150)
plt.show()