
from __future__ import unicode_literals, print_function
#Meet Robo: your friend

#import necessary libraries
import io
import random
import string
from time import time
import numpy as np
import json
import nltk
import pandas as pd
import warnings
import time
import copy
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


logs = []
story_changes = []

all_slots = {
    'account':None,
    'pin':None,
    'amountOfMoney':None,
    'yesno':None,
    'cardNumber':None,
    'cursor':0,
    'balance':1000000,
    'success':True,
    'phoneNumber':None,
    'isContinue':True,
}

slot_map_for_ban = {
    'account':'একাউন্ট নাম্বার',
    'pin':'পিন নাম্বার',
    'amountOfMoney':'টাকার পরিমান',
    'yesno':'হ্যাঁ বা না',
    'balance':'ব্যালেন্স',
    'cardNumber':'কার্ড নাম্বার',
    'success':'সফল',
    'phoneNumber':'ফোন নাম্বার'   
}
action_yesno = {{'name':'action_get_yesno','expected_intents':['affirm','deny'],'req_slots':['yesno'],'prev_step':None}}

STORIES = {                          ##utter                      action
    # 'check_balance': {'0':'utter_ask_account','1':{'inform':[{'account':76534721}]},'2':'utter_tell_account','3':{''},'3':'utter_ask_pin','3':{'inform':[{'pin':3276}]},'4':'utter_tell_balance'},
    'check_balance': {'0':{'name':'utter_ask_account'}, 
        '1':{'name':'action_get_account','expected_intents':['inform','fallback'],'req_slots':['account']},
        '2':{'name':'utter_confirm_account'},
        '3':{'name':'action_get_yesno','expected_intents':['affirm','deny'],'req_slots':['yesno'],'prev_step':'0'},
        '4':{'name':'utter_ask_pin'},
        '5':{'name':'action_get_pin','expected_intents':['inform','fallback'],'req_slots':['pin']},
        '6':{'name':'utter_confirm_pin'},
        '7':{'name':'action_get_yesno','expected_intents':['affirm','deny'],'req_slots':['yesno'],'prev_step':'4'},
        '8':{'name':'utter_tell_wait'},
        '9':{'name':'utter_tell_balance','end_story':True}},

    'bKash_transfer':{'0':{'name':'utter_ask_account'}, 
        '1':{'name':'action_get_account','expected_intents':['inform','fallback'],'req_slots':['account']},
        '2':{'name':'utter_confirm_account'},
        '3':{'name':'action_get_yesno','expected_intents':['affirm','deny'],'req_slots':['yesno'],'prev_step':'0'},
        '4':{'name':'utter_ask_phoneNumber'},
        '5':{'name':'action_get_phoneNumber','expected_intents':['inform','fallback'],'req_slots':['phoneNumber']},
        '6':{'name':'utter_confirm_phoneNumber'},
        '7':{'name':'action_get_yesno','expected_intents':['affirm','deny'],'req_slots':['yesno'],'prev_step':'4'},
        '8':{'name':'utter_ask_amountOfMoney'},
        '9':{'name':'action_get_amountOfMoney','expected_intents':['inform','fallback'],'req_slots':['amountOfMoney']},
        '10':{'name':'utter_confirm_amountOfMoney'},
        '11':{'name':'action_get_yesno','expected_int   ents':['affirm','deny'],'req_slots':['yesno'],'prev_step':'8'},
        '12':{'name':'utter_tell_wait'},
        '13':{'name':'utter_tell_success'},
        '14':{'name':'utter_tell_balance','end_story':True}},
    
}


def set_slots(slot_name,user_input,user_intent):
    # global all_slots
    # all_slots = nlu_engine.parse(user_input)

    ###TODO process user input and extract slots

    if user_intent=="affirm":
        all_slots['yesno']=True
    elif user_intent=="deny":
        all_slots['yesno']=False
    elif slot_name=="amountOfMoney":
        all_slots[slot_name] = int(user_input)
        if all_slots['balance']- int(user_input)>0:
            all_slots['balance'] =  all_slots['balance']- int(user_input)
        else:
            all_slots['success'] = False
    else:
        all_slots[slot_name] = int(user_input)


def utters_responses(to_do=None,slot_name=None):
    if(to_do=='ask'):
        if all_slots.get(slot_name) is not None and all_slots.get('yesno') is True and( slot_name == 'account' or slot_name == 'cardNumber'):
            all_slots['cursor']+=1
        else:
            print("দয়া করে "+slot_map_for_ban[slot_name]+" দিন")
            
    elif(to_do=='confirm'):
        if slot_name =='isContinue':
           print("আপনি কি আগের কাজটি সম্পন্ন করতে চান?")
        else:    
            print("আপনার "+slot_map_for_ban[slot_name]+" হল "+str(all_slots[slot_name]))
            print("এটা কি সঠিক হয়েছে?")

    elif(to_do=='tell'):
        if slot_name == 'wait':
            print("অনুগ্রহ করে কিছুক্ষণ সময় দিন")
            time.sleep(2)
            
            ##TODO: transaction in bank api

        elif slot_name == 'success':
            if all_slots['success'] == True:
                print("কাজটি সফলভাবে সম্পন্ন হয়েছে")
            else:
                print("কাজটি সফলভাবে সম্পন্ন হয়নি")
        else:        
            print("আপনার "+slot_map_for_ban[slot_name]+" হল "+str(all_slots[slot_name]))
    else:
        print(" আমি ঠিক বুঝতে পারিনি। দয়া করে আবার বলুন")


def utter_process(utter_name):
    all_slots['cursor']+=1
    utter_parts=utter_name.split('_')
    if(utter_parts[0]=='utter'):
        utters_responses(to_do=utter_parts[1],slot_name=utter_parts[2])
    else:
        utters_responses()


def action_process(full_action):
    user_response,user_intent = take_input()
    all_slots['cursor']+=1
    action_parts = full_action['name'].split('_')
    req_slots = full_action['req_slots']
    expected_intents = full_action['expected_intents']

    if user_intent not in expected_intents:
        this_story =copy.copy(current_story)
        story_changes.append(this_story)
        all_slots['cursor']=0
        run_story(user_intent)

        # utter_process('utter_confirm_isContinue')
        # action_process({'name':'action_get_yesno','expected_intents':['affirm','deny'],'req_slots':['yesno']})
        # if all_slots['yesno']==True:
        #     all_slots['cursor']-=2
        # else:
        #     run_story(user_intent)
        ###TODO: process story transitions


    ### (*) is similer to spread (...) operation in javascript
    set_slots(*req_slots,user_response,user_intent)
    if 'prev_step' in full_action:
        if all_slots['yesno']==False:
            all_slots['cursor']= int(full_action['prev_step'])
    # if(action_parts[0]=='action'):
    logs.append({'user_response':user_response,'user_intent':user_intent})


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating respons
def response(user_response):
    parsing=nlu_engine.parse(user_response)
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
        # robo_response="I am sorry! I don't understand you"
        robo_response = "fallback"
        return robo_response
    else:
        robo_response = parsing['intent']['intentName']
        return robo_response


def take_input():
    user_response = input("->")
    user_response=user_response.lower()
    user_intent = response(user_response)
    return user_response,user_intent

def dailouge_process(value):
    if value['name'].split('_')[0] == 'utter':
        utter_process(value['name'])   
        # if 'expected_intents' in value and user_intent in value['expected_intents']:
        # if value['name'].split('_')[0] == 'action':
    else:
        action_process(value)

def slot_reset():
    all_slots['cursor'] = 0
    all_slots['success'] = False
    all_slots['yesno'] = False
    all_slots['phoneNumber'] = None
    all_slots['amount'] = None
    all_slots['phoneNumber'] = None


def run_story(user_intent):
    this_story = STORIES[user_intent]
    # for key, value in STORIES.items():
    current_story['story_name'] = user_intent
    while True:
        # if not skip:
        value = this_story[str(all_slots['cursor'])]
        current_story['story_steps'] = all_slots['cursor']-1
        if 'end_story' in value:
            dailouge_process(value)
            if story_changes != []:
                story = story_changes.pop()
                utter_process('utter_confirm_isContinue')
                action_process(action_yesno)
                if all_slots['yesno']==True:
                    all_slots['cursor']=story['story_steps']
                    # print(all_slots['cursor'])
                    run_story(story['story_name'])
            else:
                # reset slots
                slot_reset()
                break
        dailouge_process(value)
        

current_story = {'story_name':None,'story_steps':None}

def alt_rasa():
    flag=True
    print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
    while(flag==True):
        user_response , user_intent = take_input()
        # print(all_slots['cursor'])
        if(user_response!='bye'):
            if(user_response=='thanks' or user_response=='thank you' ):
                flag=False
                print("ROBO: You are welcome..")

            if user_intent in STORIES:                
                run_story(user_intent)

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


alt_rasa()
            