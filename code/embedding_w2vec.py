def write_file(list):
    with open("/Users/oluwayetty1/Desktop/School/1-2/NLP/public_homework_1/data/asmsr/unigram_label.txt", mode="a") as outfile:
        for unigram in list:
            # import ipdb; ipdb.set_trace()
            outfile.write(unigram + "\n")

def split_into_bigrams(sentence: str):
    """
    :param sentence Sentence as str
    :return bigrams List of bigrams
    """
    bigrams = []
    for i in range(len(sentence)-1):
        bigram = sentence[i:i+2]
        bigrams.append(bigram)
    return bigrams

def split_into_unigram(sentence: str):
    """
    :param sentence Sentence as str
    :return bigrams List of bigrams
    """
    unigrams = []
    for i in range(len(sentence)):
        unigram = sentence[i]
        unigrams.append(unigram)
    return unigrams

with open("/Users/oluwayetty1/Desktop/School/1-2/NLP/public_homework_1/data/asmsr/label.txt") as file:
    for line in file:
        # rstrip() remove '\n' from end of line
        unigram_list_per_line = split_into_unigram(line.rstrip())
        write_file(unigram_list_per_line)
