# Neural_Network_Charity_Analysis
Module 19
# Neural Network Charity Analysis

## Analysis Overview
The purpose of this project is to use deep-learning neural networks with the TensorFlow platform in Python, to analyze and classify the success of charitable donations.\
We use the following methods for the analysis:
- preprocessing the data for the neural network model,
- compile, train and evaluate the model,
- optimize the model.

## Resources
- Data Source: [charity_data.csv](https://github.com/KdotGhai/Neural_Network_Charity_Analysis/blob/1f7070c13ea6d89b1f86bce925bcd17ac5c222b8/Resources/charity_data.csv)
- Software: Python 3.7.7, Anaconda Navigator 1.9.12, Conda 4.8.4, Jupyter Notebook 6.0.3

## Results

### Data Preprocessing
- The columns `EIN` and `NAME` are identification information and have been removed from the input data.
- The column `IS_SUCCESSFUL` contains binary data refering to weither or not the charity donation was used effectively. This variable is then considered as the target for our deep learning neural network.
- The following columns `APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT` are the features for our model.\
Encoding of the categorical variables, spliting into training and testing datasets and standardization have been applied to the features.

### Compiling, Training, and Evaluating the Model
- The model accuracy is under 75% in the initial analysis carried out as well as, the three attempts to better optimize the model. This is not a satisfying performance to help predict the outcome of the charity donations.
- This deep-learning neural network model is made of two hidden layers with 80 and 30 neurons respectively.\
The input data has 43 features and 25,724 samples.\
The output layer is made of a unique neuron as it is a binary classification.\
To speed up the training process, we are using the activation function `ReLU` for the hidden layers. As our output is a binary classification, `Sigmoid` is used on the output layer.\
For the compilation, the optimizer is `adam` and the loss function is `binary_crossentropy`.

### Optimizations
- #1
  - Keep Same data but adjust # of Hidden Layers(3) and Nuerons(2/3 of previous layers Nuerons, initial 99). This was a small alteration with the intent of keeping the data as a constant factor and see the impact of adding a 3rd hidden Layer while creating a pattern for the neurons of two-thirds of the previous layer. Overall, there was minuscule improvement without reaching the 75% threshhold(only a 0.2% improvement)
- #2
  - Adjust data(include more APPLICATION_TYPE, CLASSIFICATION), adjust # of Hidden Layers(6) and Nuerons(30 less than of previous layers Nuerons, initial 200). Here the intended goal was to show a little more leniency in including more data while "binning" `APPLICATION_TYPE` and `CLASSIFICATION`. Afterward, utilizing more hidden layers with more neurons, there was minuscule improvement(0.1% improvement)
- #3
  - Adjust data(No Binning/Bucketing and include STATUS), adjust # of Hidden Layers(4) and Nuerons( initial 150), adjust activation functions. This was a bit more extreme of an attempt at improving the model with no binning and more reasonable amount of hidden layers and nuerons with no pattern of how they decreased. In the end the improvment was minuscule(0.2% improvement)

## Summary
The deep learning neural network model did not reach the target of 75% accuracy. Considering that this target level is pretty average we could say that the model is not overfitted. Which must indicate that there's too much "noisy data" or that we must reconsider what is considered succesful since we had limited info on the charities use of the money they recieved and no info on there effective use(did they use all of the money recieved? Did they use partially? Is there cause for concern some was pocketed/redirected from intended use?)
Since we are in a binary classification situation, we could use a supervised machine learning model such as the Random Forest Classifier to combine a multitude of decision trees to generate a classified output and evaluate its performance against our deep learning model.
