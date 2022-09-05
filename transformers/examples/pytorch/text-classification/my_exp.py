import json
import matplotlib.pyplot as plt

with open('outputs/vote_100_select.json') as f:
    vote_stat = json.load(f)

votes = sorted(vote_stat.items(),key=lambda x:len(x[1]))
y = [len(i[1]) for i in votes]
x = range(len(y))
plt.plot(x,y)
plt.show()