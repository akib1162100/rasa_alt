import yaml
from yaml.loader import SafeLoader

# Open the file and load the file
intents= []
utters = []
with open('domain.yml', encoding="utf8") as f:
    data = yaml.load(f, Loader=SafeLoader)
    # print(data['intents'])
    for intent in data['intents']:
        if type(intent) is dict:
            intents.extend(intent.keys())
        else:
            intents.append(intent)
    # print(data['responses'])
    for key, value in data['responses'].items():
        dictionary ={}
        dictionary ={key:value[0]['text']}
        utters.append(dictionary)