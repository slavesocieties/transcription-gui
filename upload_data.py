import os
import requests

for folder, subfolders, files in os.walk("E:\\ssda-htr-data\\aligned"):
    base_id = folder[folder.rfind('\\') + 1:] + '-'
    for file in files:
        if 'jpg' in file:            
            line_id = file[1:]
            key = base_id + line_id
            with open(os.path.join(folder, file), 'rb') as f:
                img_data = f.read()
            headers = {"Content-Type":"image/jpeg"}
            requests.put("https://zoqdygikb2.execute-api.us-east-1.amazonaws.com/v1/ssda-htr-training/" + key, data=img_data, headers=headers)
        if 'txt' in file:
            line_id = 0
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                for line in f:
                    line_id += 1
                    key = base_id + '0' * (2 - len(str(line_id))) + str(line_id) + '.txt'
                    headers = {"Content-Type":"text/plain"}
                    requests.put("https://zoqdygikb2.execute-api.us-east-1.amazonaws.com/v1/ssda-htr-training/" + key, data=line.replace('\n', '').encode('utf-8'))