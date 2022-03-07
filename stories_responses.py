import yaml
from yaml.loader import SafeLoader

# Open the file and load the file
with open('domain.yml', encoding="utf8") as f:
    data = yaml.load(f, Loader=SafeLoader)
    print(data)