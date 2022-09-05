import json
import matplotlib.pyplot as plt

with open('outputs/vote_100_select.json') as f:
    vote_stat = json.load(f)

with open('selected_indices/vote_100_select_2.json') as f:
    selected_indices = json.load(f)

i = 1
while i<100:
    if not str(selected_indices[i]) in vote_stat:
        print(1,i,selected_indices[i])
        exit(0)
    if len(vote_stat[str(selected_indices[i])])>len(vote_stat[str(selected_indices[i-1])]):
        print(2, i,selected_indices[i])
        exit(0)
    i += 1

print(len(vote_stat[str(selected_indices[99])]))
