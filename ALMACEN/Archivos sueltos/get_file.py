import json
def get_file(path_file):
    with open(path_file) as f:
        data = json.load(f)
        
