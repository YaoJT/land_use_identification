import matplotlib.pyplot as plt
import numpy as np

ff = open('4_29_home_mac_no_water/score.txt').readlines()
print(len(ff))
ff = ff[0].split('},')
ff = [x.replace('{','').replace('}','') for x in ff]
ff = [x.split(',') for x in ff]
print(ff[0])
mean_x = [float(x[-1].split(':')[1]) for x in ff]
##plt.plot([mean_x[i+2000]-mean_x[i] for i in range(2000)])
##plt.plot([np.mean(mean_x[i]) for i in range(2000,4000)])
plt.plot([np.mean(mean_x[i-2000:i]) for i in range(2000,len(mean_x))])
plt.ylim(0.5,1)
plt.plot([0 for x in range(2000)])
plt.title(np.mean(mean_x[-2000:]))
plt.show()
