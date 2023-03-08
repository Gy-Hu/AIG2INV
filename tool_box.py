
# log path: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/tmp/
import os
import re
import base64
from IPython.display import Image, display
from PIL import Image
import matplotlib.pyplot as plt
import requests
import io
import json

from natsort import natsorted


def walkFile(dir):
    files = None
    for root, _, files in os.walk(dir):
        files = natsorted(files)
        files = [os.path.join(root, f) for f in files]
    return files


def clean_trivial_log():
    log_path = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/tmp/"
    # get all the .log file under log_path
    log_files = os.listdir(log_path)
    # filter out the .log file
    log_files = [
        log_file for log_file in log_files if log_file.endswith(".log")]
    # try to clean the trivial log if line number is less than 20
    for log_file in log_files:
        with open(os.path.join(log_path, log_file), "r") as f:
            lines = f.readlines()
            if len(lines) < 20:
                # use trash command to delete the file
                os.system(f"trash {os.path.join(log_path, log_file)}")
                print(f"Delete {log_file} successfully!")


def calculate_pickle_number():
    #json_path = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc07_0_1_2_3_4/bad_cube_cex2graph/json_to_graph_pickle/"
    json_path = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_hwmcc07_almost_complete/bad_cube_cex2graph/json_to_graph_pickle/"
    # get all files in the json_path
    for root, _, files in os.walk(json_path):
        files = [os.path.join(root, f) for f in files]
    # remove the _{number}.pkl and use set to remove duplicate in files list
    # Create the regular expression
    regex = re.compile(r'(.*)_[0-9]+\.pkl$')
    # Apply the regular expression to the array elements
    output_array = [regex.match(element)[1] for element in files]
    # Remove duplicate elements from the output array
    output_array = list(set(output_array))
    print("The number of pickle files is: ", len(output_array))

# draw mermaid


def json2mermaid(json_path=None):
    with open(json_path) as f:
        json_obj = json.loads(f.read())
    if json_path == None: 
        json_obj = json.loads('''
       [    
       {        
            "data": {            
                    "application": "not",            
                    "id": 5,            
                    "to": {                
                        "children_id": [18]
                    },
                    "type": "node"
                }
            },
        {
        "data": {
            "application": "and",
            "id": 6,
            "to": {
                "children_id": [
                    19,
                    20
                ]
            },
            "type": "node"
        }
    },
    {
        "data": {
            "application": "i10",
            "id": 18,
            "type": "variable"
        }
    },
    {
        "data": {
            "application": "i11",
            "id": 19,
            "type": "variable"
        }
    },
    {
        "data": {
            "application": "i12",
            "id": 20,
            "type": "variable"
        }
    }
]

    ''')
    graph = 'graph TB\n'
    for obj in json_obj:
        data = obj['data']
        if data['type'] == 'node':
            children_id = data['to']['children_id']
            # for child in children_id, {data["id"]}[{data["application"]}] --> {data["children_id"]}[data["children_application"]]
            # id[application] --> children_id[children_application]
            for child_id in children_id:
                child_data = next((d for d in json_obj if d["data"]["id"] == child_id), None)
                if child_data:
                    child_application = child_data["data"]["application"]
                    graph += f'{data["id"]}[{data["id"]}:{data["application"]}] --> {child_id}[{child_id}:{child_application}]\n'
    #graph += '\n'
    graphbytes = graph.encode("ascii")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    img_data = requests.get(f'https://mermaid.ink/img/{base64_string}').content
    with open('mermaid.jpg', 'wb') as f:
        f.write(img_data)


if __name__ == "__main__":
    # clean_trivial_log()
    # walkFile()
    # calculate_pickle_number()
    json2mermaid("/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset_20230106_014957_toy/bad_cube_cex2graph/expr_to_build_graph/nusmv.syncarb5^2.B/nusmv.syncarb5^2.B_7.json")
