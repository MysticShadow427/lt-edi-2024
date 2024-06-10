import pandas as pd
import numpy as np
import re

def remove_newline_pattern(sentence):
    # Remove '\n' pattern
    sentence = re.sub(r'\n', '', sentence)
    return sentence

# Function to remove numbers and URLs from a sentence
def remove_numbers_and_urls(sentence):
    # Remove numbers
    sentence = re.sub(r'\d+', '', sentence)
    # Remove URLs
    sentence = re.sub(r'http[s]?://\S+', '', sentence)
    return sentence

# Define a function to remove emojis
def remove_emojis(text):
    if isinstance(text, str):
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F700-\U0001F77F"  # alchemical symbols
                                u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                u"\U00002702-\U000027B0"  # Dingbats
                                u"\U000024C2-\U0001F251" 
                                "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    else:
        return text

# Function to remove pattern occurrences of '@Username' from a sentence
def remove_pattern(sentence):
    return re.sub(r'@[\w]+', '', sentence)