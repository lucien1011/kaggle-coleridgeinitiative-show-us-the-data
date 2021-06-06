import os
import re
import json
import string

train_files_path = 'input/train'
test_files_path = 'input/test'
sec_divider = '|||||||'

def clean_text(s):
    """
    This function is essentially clean_text without lowercasing.
    """
    s = re.sub('[^A-Za-z0-9]+', ' ', str(s)).strip()
    s = re.sub(' +', ' ', s)
    return s

def json_to_text(filename, train_files_path=train_files_path, output='text'):
    json_path = os.path.join(train_files_path, (filename+'.json'))
    headings = []
    contents = []
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for data in json_decode:
            headings.append(data.get('section_title'))
            contents.append(data.get('text'))
    
    #all_headings = sec_divider.join(headings)
    #all_contents = sec_divider.join(contents)
    
    if output == 'text':
        return all_contents
    elif output == 'head':
        return all_headings

def json_to_list(filename, train_files_path=train_files_path, output='text'):
    json_path = os.path.join(train_files_path, (filename+'.json'))
    headings = []
    contents = []
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for data in json_decode:
            headings.append(data.get('section_title'))
            contents.append(data.get('text'))
    
    if output == 'text':
        return contents
    elif output == 'head':
        return headings

def text_cleaning(text):
    text = ''.join([k for k in text if k not in string.punctuation])
    text = re.sub('[^A-Za-z0-9]+', ' ', str(text).lower()).strip()
    text = re.sub(' +', ' ', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text

