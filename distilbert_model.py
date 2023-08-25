import ktrain
import os


os.environ['CUDA_VISIBLE_DEVICES'] = "-1" # use only CPU in prediction

# Creating a function that will link to your web API using Flask and loading in the model to do the job/ prediction

predictor = ktrain.load_predictor('/home/sedeba19/Sentiment_Classification_using_DistilBERT/BERT DistilBERT Project/distilBERT-model')

def get_prediction(x):
    sent = predictor.predict([x])
    return sent[0]