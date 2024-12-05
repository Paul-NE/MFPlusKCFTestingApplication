import pandas as pd
from typing import TextIO


def postprosess_table(exel_file: TextIO):
    
    print(pd.read_excel(exel_file))
    

def main():
    with open("/home/poul/temp/test.xlsx", "r") as exel_file:
        # "test1"
        postprosess_table(exel_file)

if __name__=="__main__":
    pass