The data folder contains the data to train the model.

## Training data
1.dat to 3.dat simulate data that entered on different time slices.
The format of the data is:
```
[Doc_1 name] [token_1] [token_2] ... [token_N]
     :
     :
[Doc_M name] [token_1] [token_2] ... [token_N]
```
Each line is a document and it is assumed, that data preprocessing has been performed already (removal of stop words, stemming,...).

## Sentiment Lexicon
The algorithm needs prior information on the sentiment-polarity of words. The sentiment lexicon used here is MPQA and has the following format:
````
[word]	[neu prior prob.] [pos prior prob.] [neg prior prob.]
````
