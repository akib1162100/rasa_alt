
from __future__ import unicode_literals, print_function
#Meet Robo: your friend

#import necessary libraries
import io
import random
import string
import numpy as np
import json
import nltk
import pandas as pd
import warnings

#for snips

from venv import create # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN
from nltk.stem import WordNetLemmatizer
from stories_responses import *

warnings.filterwarnings('ignore')

nltk.download('popular', quiet=True) # for downloading packages

with io.open("dataset.json") as f:
    sample_dataset = json.load(f)

nlu_engine = SnipsNLUEngine(config=CONFIG_EN)
nlu_engine = nlu_engine.fit(sample_dataset)
# uncomment the following only the first time
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only


#Reading in the corpus
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


log = None
all_slots = {
    'account':None,
    'pin':None,
    'amount_of_money':None,
    'yes_no':False
}

slot_map_for_ban = {
    'account':'একাউন্ট নাম্বার',
    'pin':'পিন নাম্বার',
    'amount_of_money':'টাকার পরিমান',
    'yes_no':'হ্যাঁ বা না'   
}

STORIES = {                          ##utter                      action
    # 'check_balance': {'0':'utter_ask_account','1':{'inform':[{'account':76534721}]},'2':'utter_tell_account','3':{''},'3':'utter_ask_pin','3':{'inform':[{'pin':3276}]},'4':'utter_tell_balance'},
    'check_balance': {'0':{'name':'utter_ask_account'},
        '1':{'name':'action_get_account','expected_intents':['inform'],'req_slots':['account'],'prev_step':None},
        '2':{'name':'utter_tell_account'},
        '3':{'name':'action_yes_no','expected_intents':['affirm','deny'],'req_slots':['yes_no'],'prev_step':'1'},
        '4':{'name':'utter_ask_pin'},
        '5':{'name':'action_get_pin','expected_intents':['inform'],'req_slots':['pin'],'prev_step':None},
        '6':{'name':'utter_tell_pin'},
        '7':{'name':'action_yes_no','expected_intents':['affirm','deny'],'req_slots':['yes_no'],'prev_step':'7'},
        '8':{'name':'utter_tell_balance'}}
}


def utters_responses(to_do=None,slot=None):
    if(to_do=='ask'):
        print("দয়া করে "+slot_map_for_ban[slot]+" দিন")
    elif(to_do=='tell'):
        print("আপনার "+slot_map_for_ban[slot]+" হল "+str(all_slots[slot]))
        print("এটা কি সঠিক হয়েছে?")
    else:
        print(" আমি ঠিক বুঝতে পারিনি। দয়া করে আবার বলুন")


def utter_process(utter_name):
    utter_parts=utter_name.spilt('_')
    if(utter_parts[0]=='utter'):
        user_response(to_do=utter_parts[1],slot=utter_parts[2])
    else:
        user_response()

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    parsing=nlu_engine.parse(user_response)
    print(parsing)
    probability=parsing['intent']['probability']

    # robo_response=''
    # sent_tokens.append(user_response)
    # TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    # tfidf = TfidfVec.fit_transform(sent_tokens)
    # vals = cosine_similarity(tfidf[-1], tfidf)
    # idx=vals.argsort()[0][-2]
    # flat = vals.flatten()
    # flat.sort()
    # req_tfidf = flat[-2]
    if(float(probability)<0.60):
        robo_response="I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = parsing['intent']['intentName']
        return robo_response


flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        
        if (user_response=='check balance'):
            story_intents = []
            slots = []
            this_story = STORIES['check_balance']
            for key, value in STORIES.items():
                if value['name'].split('_')[0] == 'utter':
                    utter_process(value['name'])


        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                #sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")    
        
        