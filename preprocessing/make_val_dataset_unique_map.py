import json
import pickle

input_path = "storage/input/rcdataset/data_sets.json"
output_path = "storage/input/rcdataset/data_sets_unique_map.p"

allow_word_list = [
        "study",
        "Study",
        "studies",
        "Studies",
        "statistics",
        "Statistics",
        "project",
        "Project",
        "projects",
        "Projects",
        "program",
        "Program",
        "programs",
        "Programs",
        "National",
        "national",
        "Cohort",
        "cohort",
        "Report",
        "Reports",
        "report",
        "reports",
        "Data",
        "data",
        ]

def get_unique_title(ds):
    filter_ds = [d for d in ds if all(not i.isdigit() for i in d) and "(" not in d and ")" not in d]
    if len(filter_ds) == 0: return None
    dsorted = sorted(filter_ds,key=lambda x: len(x),reverse=True)
    return dsorted[0]

if __name__ == "__main__":

    f = json.load(open(input_path,"r"))
    out_dict = {}
    for d in f:
        if d['mention_list']:
            d['unique_dataset_name'] = get_unique_title(d['mention_list'])
            if not d['unique_dataset_name']:
                d['unique_dataset_name'] = d['title']
        else:
            d['unique_dataset_name'] = d['title']

        for m in d['mention_list']:
            if not any([w in m for w in allow_word_list]):
                continue
            if m not in out_dict:
                out_dict[m] = d

    pickle.dump(out_dict,open(output_path,"wb"))
