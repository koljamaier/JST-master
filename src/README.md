This readme provides a brief overview on the most important classes in the project. For further details refer to the documentation in [Help](https://github.com/koljamaier/JST-master/tree/djst/Help) 

#### dataset.cpp
This class manages several document instances. 

#### document.cpp
Saves word ids and their corresponding prior-senti-labels for one single document. In contrast, dataset saves several instances of document

#### Inference.cpp //TODO: change name
This class implements the dJST model. Several JST models (model) are trained and combined over time.

#### model.cpp
Offers parameters and functions to train a single time slice JST model. 

#### utils.cpp
This class is used to 
1. parse the command line arguments and 
1. read out the parameters specified in the corresponding .properties file. 
