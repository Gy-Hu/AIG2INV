
# log path: /data/guangyuh/coding_env/AIG2INV/AIG2INV_main/log/tmp/
import os

if __name__ == "__main__":
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
                