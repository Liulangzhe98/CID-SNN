import random
from pathlib import Path

import glob
import math
import json   
import os
import re


from PIL import Image

def log_validator(root: Path):
    files = glob.glob(f"{root}/*")
    files.sort(key=os.path.getmtime)


    for log_file in files:
        with open(log_file, "r") as f:
            print(log_file)
            all_lines = f.read()

            if re.findall(r"slurmstepd", all_lines) != []:
                print("ERROR")
                continue

            # print(all_lines)
            a = re.findall(r"HYPER.*", all_lines)
            # b = re.findall(r"(Prec.*[\d\.]*%).*\n(Recall: [\d\.]*%)", all_lines)
            b = re.findall(r"(Prec.*|Recall.*|Fscore.*|Acc.*)%", all_lines)
            print(a)
            print(b)
            
        print()



if __name__ == "__main__":
    log_validator(Path("logs"))
