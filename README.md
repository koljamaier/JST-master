# Dynamic Joint Sentiment-Topic Model (dJST)
This project aims to extend the Joint Sentiment-Topic Model ([JST](https://github.com/linron84/JST)) by a dynamic time component.
With this, sentiment-topics can be analyzed over time to do trend analytics.

## Getting Started
To compile the project type `make` in a shell. Alternatively you can import the project into Visual Studio.

### Estimation (JST)
For training on a single dataset execute:
`jst -est -config YOUR-PATH/train.properties`
This is equivalent to execute JST.

### Inference/Continnous Estimation (dJST)
To continuously train on new data execute:
`jst -inf -config YOUR-PATH/test.properties`

## Output
The trained parameters/distributions will be written out in the following files:
* `<iter>.others`
Holds summaries on the corpus statistics, like how many docs and words have been discovered.
Also the training parameter details will be displayed (how many iterations, topics, senti-labels)
* `<iter>.pi`
Contains the per-document sentiment distributions
* `<iter>.phi`
Contains the sentiment specific topic-word distributions
* `<iter>.theta` 
Contains the per-document sentiment specific topic proportions. For each topic the sentiment distribution is shown
* `<iter>.tassign`
Contains the sentiment label and topic assignments for words in training data. After sampling each word was assigned to a topic- and sentiment-label

## Data
The data input format should be like the following:
```
[Doc_1 name] [token_1] [token_2] ... [token_N]
     :
     :
[Doc_M name] [token_1] [token_2] ... [token_N]
```
Each line is a document and it is assumed, that data preprocessing has been performed already (removal of stop words, stemming,...).

The algorithm needs prior information on the sentiment-polarity of words. The sentiment lexicon used here is MPQA and has the following format:
````
[word]	[neu prior prob.] [pos prior prob.] [neg prior prob.]
````

## Authors
* **Chenghua Lin** - *Initial work JST* - [linron84](https://github.com/linron84)
* **Kolja Maier** extension to dJST

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* Fur further documentation on classes and functions see /Help

## References
* *Lin, He, Yulan and Lin, Chenghua and Gao, Wei and Wong, Kam-Fai Dynamic joint sentiment-topic model, ACM Transactions on Intelligent Systems and Technology, 2013.*
* *Lin, C., He, Y., Everson, R. and Reuger, S. Weakly-supervised Joint Sentiment-Topic Detection from Text, IEEE Transactions on Knowledge and Data Engineering (TKDE), 2011.*
* *Lin, C. and He, Y. Joint Sentiment/Topic Model for Sentiment Analysis, In Proceedings of the 18th ACM Conference on Information and Knowl- edge Management (CIKM), Hong Kong, China, 2009.*

