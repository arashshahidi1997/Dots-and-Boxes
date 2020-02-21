import numpy as np
import matplotlib.pyplot as plt

a_num = 2
o_num = 0
turn = 'second'
folder = ''
file = turn + '_learner' + str(a_num)+'against_opp'+str(o_num)+'.npy'
data = np.load(folder+file)
fig = plt.figure()
ax = fig.add_subplot(111)
title = 'learner ' + str(a_num) + ' as '+turn+' player against static agent ' + str(o_num)
fig.suptitle(title)
print(data)
length = np.size(data)

for i in range(length):
    data[i] = (data[i]-data[0])/(i+1)

ax.plot(np.arange(length)+1, data)
ax.xlabel = 'number of matches'
ax.ylabel = 'total reward collected'
fig.savefig(title + '.png')