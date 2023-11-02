import json
import os
print(os.getcwd())
# def load_config_dict(filepath='./config_test.json'):
def load_config_dict(filepath='./config/config.json'):
    with open(filepath) as json_file:
        config_dict = dict(json.load(json_file))
    return config_dict