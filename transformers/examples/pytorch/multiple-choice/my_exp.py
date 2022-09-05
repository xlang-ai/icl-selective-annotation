import json
import matplotlib.pyplot as plt

with open('outputs/vote_20_select.json') as f:
    vote_stat = json.load(f)

votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
y = [len(i[1]) for i in votes]
x = range(len(y))
plt.plot(x,y)
plt.title('HellaSwag')
plt.xlabel('example index')
plt.ylabel('number of supporters')
plt.show()