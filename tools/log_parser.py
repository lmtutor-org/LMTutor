import json, re
from pathlib import Path
import time
from tqdm import tqdm


def log_loader(log_dir, data={}):
    conv_file_name_pattern = re.compile(r'\d{4}-\d{2}-\d{2}-conv\.json')
    conv_files = [file for file in log_dir.iterdir() if conv_file_name_pattern.match(file.name)]

    for filename in tqdm(conv_files, desc="loading convs"):
        for retry in range(5):
            try:
                lines = open(filename).readlines()
                break
            except FileNotFoundError:
                time.sleep(2)
            
        for l in lines:
            row = json.loads(l)
            if "user_PID" in row.keys():
                conv_info = {"user_name": row.get("user_name"), 
                             "user_answer": row.get("user_answer"), 
                             "user_reason": row.get("user_reason"),
                             "conv": row["state"]["messages"]}

                if row["user_PID"] in data.keys():
                    data[row["user_PID"]].append(conv_info)
                    data[row["user_PID"]][0]+=1
                else:
                    data[row["user_PID"]] = [1, conv_info]
    
    return data

# def parse_conv_info(data):
#     user_info = {}
#     for row in data:
#         user_info[row["user_name"]] = {"num_submissions": row["user_answer"], 
#                                        "user_reason": row["user_reason"]}

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def main():
    log_dir_0 = Path("../lmtutor_dsc250_logs/lmtutor_v0.0_logs")
    log_dir_1 = Path("../lmtutor_dsc250_logs/lmtutor_v0.1_logs")
    data = log_loader(log_dir_0)
    data = log_loader(log_dir_1, data)
    save_to_json(data, "data.json")
    pass


if __name__ == "__main__":
    main()