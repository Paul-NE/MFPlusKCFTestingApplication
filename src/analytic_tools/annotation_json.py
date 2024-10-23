import json


class AnnotationJson:
    pass


def show_lvl_dict(data:dict|list ,lvl:int=0):
    print(f"{'\t'*lvl}{{")
    for key, value in data.items():
        print(f"{'\t'*(lvl+1)}{key}: {type(value).__name__}")
        if isinstance(value, dict) or isinstance(value, list):
            show_lvl(value, lvl+1)
            continue
        print(f"{'\t'*(lvl+2)}{value}")
    print(f"{'\t'*lvl}}}")

def show_lvl_list(data:dict|list ,lvl:int=0):
    print(f"{'\t'*lvl}[")
    if len(data) > 3:
        data = data[:3]
        data.append("...")
    for item in data:
        if isinstance(item, dict) or isinstance(item, list):
            show_lvl(item, lvl+1)
            continue
        print(f"{'\t'*(lvl+1)}{item}")
    print(f"{'\t'*lvl}]")

def show_lvl(data:dict|list ,lvl:int=0):
    operations = {
        "list": show_lvl_list,
        "dict": show_lvl_dict
    }
    operations[type(data).__name__](data, lvl)

if __name__=="__main__":
    json_mini = r"/mnt/2terdisk/downloads/project-3-at-2024-01-25-15-53-eba11cd7.json"
    json_macro = r"/mnt/2terdisk/downloads/project-12-at-2024-05-22-11-04-36153850.json"
    with open(json_mini, "r") as json_file:
        show_lvl(json.load(json_file)[0])