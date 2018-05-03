# Naive Bayes Classifier
Applying Naive Bayes Classifier for NLP based classification of reviews.

This implementation uses word counts for each class of review (Pos/Neg True/Fake) as features and uses log probabilities to build a model file. This model file produces emmission and transition probabilities which are used for predictions in test dataset.


## Usage

    python3 nblearn.py <inputfile-traindata>
    python3 nbclassify.py <inputfile-testdata>
    python3 nbevaluate.py <inputfile-testkeys>
    
## Results

#### Reference F1 Scores
- Pos F1=0.93
- Neg F1=0.92
- True F1=0.89
- Fake F1=0.89
- Mean F1 = 0.9078

#### Model F1 Scores
- Pos F1=0.91
- Neg F1=0.98
- True F1=0.87
- Fake F1=0.93
- Mean F1 = 0.9171
