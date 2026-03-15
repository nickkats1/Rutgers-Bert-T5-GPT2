"""Convert text into polarity and sentiments"""

from textblob import TextBlob


def polarity(text: str) -> float:
    """Convert Text into polarity scores.
    
    Args:
        text: input text to be converted into polarity scores.
        
    Returns:
        float: a value ranging from -1 -> 1 to define the polarity of the text.
    """
    return TextBlob(text).polarity



def sentiment(label: float) -> int:
    """Input polarity scores to then get discrete sentiment value
    returned.
    
    Args:
        text: input polarity score.
    
    Returns:
        sentiment: A value of either 1, 0, or -1 depending on the polarity
    """
    if label == 0:
        return "Neutral"
    elif label < 0:
        return "Negative"
    else:
        return "Positive"
    

    
    


