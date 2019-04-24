outfile= '/path/to/directory/file'
infile = '/path/to/directory/file'

# writes each ngram per line into a new file
def write_file(list):
    with open(outfile, mode="a") as o_file:
        for ngram in list:
            o_file.write(ngram + "\n")

# function to split a file into bigram per line
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

# function to split a file into unigram per line
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

with open(infile) as file:
    for line in file:
        # rstrip() remove '\n' from end of line
        unigram_list_per_line = split_into_unigram(line.rstrip())
        write_file(unigram_list_per_line)
