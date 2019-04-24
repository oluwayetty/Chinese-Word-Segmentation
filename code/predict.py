from argparse import ArgumentParser
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import precision, split_input_to_unigram, map_label_to_character

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()

def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    print("Loading files from Resources")
    json_file = open(resources_path+'/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(resources_path+ '/weights.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    sgd = optimizers.SGD(lr=0.04, momentum=0.95)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[precision])

    with open(input_path, 'r') as file:
        for line in file:
            line = line.rstrip()
            unigram_list = split_input_to_unigram(line)
            tokenizer = Tokenizer(num_words= 128)
            tokenizer.fit_on_texts(line)

            labels = []
            for unigram in unigram_list:
                sequences = tokenizer.texts_to_sequences(unigram)
                X = pad_sequences(sequences, maxlen=50)
                label = loaded_model.predict_classes(X)
                labels.append(label[0])

            with open(output_path, mode="a") as outfile:
                outfile.write(''.join(map_label_to_character(labels)) + "\n")

    print("Output saved in " + output_path)

if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
