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
