![Model Architecture](https://github.com/oluwayetty/Chinese-Word-Segmentation/blob/master/model.jpg)

## Problem 
Implemented a state-of-the-art word segmenter model in Tensorflow/Keras using Chinese characters. The image below shows the summary of this project, the input, and the corresponding output. The BIES format is a way to encode the output of a word segmenter model. There are 4 classes the model has to predict. 

## Example in the English world
Input: This is a NLP project AND Output: BIIE BE S BIE BIIIIIE
● B means Beginning of a word
● I mean In the middle of a word
● E means End of a word
● S means single, more examples of this are ".", "a", "," etc.

![](https://github.com/oluwayetty/word-sense-disambiguition/blob/master/bies.jpg)

## Dataset Description 
I made use of the datasets which can be downloaded [here](http://sighan.cs.uchicago.edu/bakeoff2005/). The full dataset contains four smaller datasets:
● AS (Traditional Chinese)
● CITYU (Traditional Chinese)
● MSR (Simplified Chinese)
● PKU (Simplified Chinese)
Note that you are responsible to convert the Traditional Chinese datasets to Simplified Chinese by Installing [HanziConv](https://pypi.org/project/hanziconv/0.3/) and run the following command: ```hanzi-convert -s infile > outfile```

## Repository skeleton
```
- code               # this folder contains all the code related to this project
- resources          # this folder contains the best saved model and its weights 
- README.md          # this file
- Homework_2_nlp.pdf # the slides for the course homework instruction
- report.pdf         # my report which basically analyzed the code and the results obtained.
```

Link to paper - (https://www.aclweb.org/anthology/D18-1529/)
