import json
import pickle

input_path = "storage/input/rcdataset/data_sets.json"
output_path = "storage/input/rcdataset/data_sets_unique_map.p"

f = json.load(open(input_path,"r"))

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
        ]

out_dict = {}
for d in f:
    for m in d['mention_list']:
        if not any([w in m for w in allow_word_list]):
            continue
        if m not in out_dict:
            out_dict[m] = d
pickle.dump(out_dict,open(output_path,"wb"))
