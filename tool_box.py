
# log path: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/tmp/
import os
import re

from natsort import natsorted

def walkFile(dir):
    files = None
    for root, _, files in os.walk(dir):
        files = natsorted(files)
        files = [os.path.join(root,f) for f in files]
    return files

def clean_trivial_log():
    log_path = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/tmp/"
    # get all the .log file under log_path
    log_files = os.listdir(log_path)
    # filter out the .log file
    log_files = [log_file for log_file in log_files if log_file.endswith(".log")]
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
    json_path = "/data/guangyuh/coding_env/AIG2INV/AIG2INV_main/dataset/bad_cube_cex2graph/json_to_graph_pickle/"
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

        

if __name__ == "__main__":
    #clean_trivial_log()
    #walkFile()
    calculate_pickle_number()