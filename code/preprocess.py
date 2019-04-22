import os
import sys

# a function to write each stripped line back into a new input file
def write_stripped_line(string, filepath):
    with open(filepath, "a") as file:
        file.write(string)

# some chinese files contains different representations of spaces,
# this function cleans them up for each peculiar case
def clean_up(line):
    if '\ufeff' in line:
        string = line.replace("\n", "").replace('\ufeff', '').split(' ')
    elif '\u3000' in line:
        string = line.replace("\n", "").split("\u3000")
    else:
        # removing the '\n' for easy assignment of labels to each line
        string = line.replace("\n", "").split(' ')
    return string

if len(sys.argv) > 1:
    argument = sys.argv[1]
    filename = os.path.basename(argument).split('.')[0]
    filepath, file_extension = os.path.splitext(argument)

    #creating directories for our new input and label files
    input_directory = os.path.join(os.path.dirname(filepath), r'input')
    label_directory = os.path.join(os.path.dirname(filepath), r'label')

    # formatting the input and label txt files naming convention
    label_filepath = label_directory + "/" + filename+ "_label" + ".txt"
    input_filepath = input_directory + "/" + filename + ".txt"

    # creating individual folders for input and label files
    folders = [input_directory, label_directory]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    with open(argument) as f:
        for line in f:
            # This section takes each line in the file and create an input
            # file i.e the one without the spaces.
            stripped = line.replace(' ','').replace('\u3000', '').replace('\ufeff', '')
            write_stripped_line(stripped, input_filepath)

            # This section takes each line in the file and create a label
            # file i.e the translation into BIES format.
            splitted_version = clean_up(line)
            for string in splitted_version:
                index = splitted_version.index(string)
                if len(string) == 1:
                    if index != splitted_version[-1]:
                        splitted_version[index] = 'S'
                    if index == splitted_version[-1]:
                        splitted_version[index] = 'E'
                elif len(string) == 2:
                    splitted_version[index] = 'BE'
                elif len(string) >= 3:
                    splitted_version[index] = "".join(('B', string[1:].replace(string[1:], "I"*(len(string[1:]) - 1)), 'E'))
            string_label = ''.join(map(str, splitted_version))
            write_stripped_line(string_label+"\n", label_filepath)
else:
    print("You must pass a valid file path to the script")
