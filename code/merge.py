filenames = [
             '/Users/oluwayetty1/Desktop/School/1-2/NLP/public_homework_1/simplified_data/training/input/msr_training.txt',
             '/Users/oluwayetty1/Desktop/School/1-2/NLP/public_homework_1/simplified_data/training/input/as_training.txt']

with open('/Users/oluwayetty1/Desktop/School/1-2/NLP/public_homework_1/data/asmsr/input.txt', 'w+') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
