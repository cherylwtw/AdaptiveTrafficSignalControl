import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from io import StringIO

# copy results here
s = StringIO("""     AverageQueueLength     AverageWaitingTime
Q_learning,RLTSC-2     18.6611   144.0997
Q_learning,RLTSC-1    18.7488  156.199
Dyna_Q,RLTSC-2     20.43   179.9302
Dyna_Q,RLTSC-1    20.2481  175.0946
Fixed_Control    22.6061  186.0141""")

df = pd.read_csv(s, index_col=0, delimiter=' ', skipinitialspace=True)

fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes

ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.25

df.AverageQueueLength.plot(kind='bar', color='red', ax=ax, width=width, position=1)
df.AverageWaitingTime.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_xticklabels( ['Q learning\nRLTSC-2','Q learning\nRLTSC-1','Dyna Q\nRLTSC-2','Dyna Q\nRLTSC-1','Fixed Control'], rotation=45 )
plt.subplots_adjust(bottom=0.18, top=0.9)


ax.set_ylabel('Average Queue Length (Num of Vehicles in Queue)', color='red')
ax2.set_ylabel('Average Waiting Time (Seconds)', color='blue')

for p1 in ax.patches:
    ax.annotate('%.1f' % p1.get_height(),  (p1.get_x() * 0.95, p1.get_height() * 1.005))

for p2 in ax2.patches:
    ax2.annotate('%.1f' % p2.get_height(),  (p2.get_x() * 1.005, p2.get_height() * 1.005))

plt.title('Average Queue Length and Waiting Time Comparison')

plt.show()
