import os
import json
import pandas as pd


def data_prepr(result):
     result= result.split(',')[0].replace(" ", "")
     result= result.replace("[", "")
     result= result.replace("]", "")
     return result

def find_word_order(text, word_list):
    #print(text)
    # Find the index of each word in the text
    indices = {word: text.find(word) for word in word_list}
    # Filter out words that were not found (find returns -1 when it doesn't find the word)
    found_indices = {word: index for word, index in indices.items() if index != -1}
    # Sort the words by their index
    sorted_words = sorted(found_indices, key=found_indices.get)
    return sorted_words

def extract_result(result):
    # The ground truth is the last part of the file path before the extension
    if type(result)==type(''):
        result= find_word_order(result,['staff', 'project', 'other', 'faculty', 'department', 'student', 'course'])
        #print('result',result)
        if len(result)>0:
            return  result[0]
        else:
            return 'other'
    else:
        return ''

def format_predictions(directory_path):
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.json'):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                for item in file_data:
                    data.append((os.path.splitext(os.path.basename(item["Titulo"]))[0], extract_result(item["Resultado"])))
            df = pd.DataFrame(data, columns=["file_name", "predicted_class"])
            df.to_csv(os.path.join(directory_path, file_name.replace('.json', '.csv')), index=False)

format_predictions('./resultados/')