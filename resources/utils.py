class Util:
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision. Computes the precision, a
        metric for multi-label classification of how many selected items are
        relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def split_input_to_unigram(line):
        """
        :param sentence Sentence as str
        :return bigrams List of bigrams
        """
        unigrams = []
        line = line.rstrip()
        for i in range(len(line)):
            unigram = line[i]
            unigrams.append(unigram)
        return unigrams

    # 0,1,2,3 in the label exactly match B,E,I,S
    def map_label_to_character(array):
        for x in array:
            if x == 0:
                array[array.index(x)] = "B"
            elif x == 1:
                array[array.index(x)] = "E"
            elif x == 2:
                array[array.index(x)] = "I"
            elif x == 3:
                array[array.index(x)] = "S"
        return "".join(array)
