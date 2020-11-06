#pip install flair, requires PyTorch use command from https://pytorch.org/
from flair.data import Sentence
from flair.models import TextClassifier
analyzer = TextClassifier.load('en-sentiment')
phrase = Sentence("I love cheese")
analyzer.predict(s)
sentiment = phrase.labels
sentiment