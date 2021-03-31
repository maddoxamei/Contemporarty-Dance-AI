from lib.phrase_classifier_dependencies import *

def _pre_process_sentence(raw):
    """ Remove hyperlinks and markup """
    result = re.sub("<[a][^>]*>(.+?)</[a]>", 'Link.', raw)
    result = re.sub('&gt;', "", result)
    result = re.sub('&#x27;', "'", result)
    result = re.sub('&quot;', '"', result)
    result = re.sub('&#x2F;', ' ', result)
    result = re.sub('<p>', ' ', result)
    result = re.sub('</i>', '', result)
    result = re.sub('&#62;', '', result)
    result = re.sub('<i>', ' ', result)
    result = re.sub("\n", '', result)
    return result

def _classify_sentence(sentence, analyzer):
    """ Calculate the sentiment score(s) for a given sentence.
    
    :param sentence: the single sentence to be classified
    :type str
    :param analyzer: the flair text classifier
    :type flair.models.TextClassifier
    :return: cooresponding sentiment 
    :type float
    """
    phrase = Sentence(sentence)
    analyzer.predict(phrase)
    sentiment = phrase.labels[0].to_dict()
    if(sentiment['value']=='POSITIVE'):
        return sentiment['confidence']
    else:
        return -sentiment['confidence']
    
def load_analyzer():
    return TextClassifier.load('en-sentiment')

def get_sentiment(text, analyzer=None):
    """ Calculate the sentiment score(s) for the text by summing the scores of the individual sentences which make it up.
    
    :param raw: the original phrase to analyze
    :type str
    :return: a list with the sentiment scores of each individual sentence and the overall score based on the summation of the list
    :type tuple
    """
    if(not analyzer):
        analyzer = load_analyzer()
    scores = []
    for s in [sent for sent in text.split(". ")]: 
        scores.append(_classify_sentence(s, analyzer))
    overall_score = round(sum(scores), 3)
    return scores, overall_score