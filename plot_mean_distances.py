

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# This script plots the results of our tests

# in seconds
window_sizes = [30, 20, 10, 4]


# in milliseconds
phasenet_distances_p = [44.78, 47.67, 42.47, 42.20]

ross_distances_p = [244.30, 210.10, 99.34, 40.80]

cospy_distances_p = [36.04, 35.72, 37.17, 34.28]



phasenet_distances_s = [60.97, 74.05, 63.43, 60.45]

ross_distances_s = [222.20, 208.70, 97.92, 49.45]

cospy_distances_s = [49.95, 49.83, 48.83, 46.11]



plt.plot(window_sizes, cospy_distances_p, label='Cospy')

plt.plot(window_sizes, phasenet_distances_p, label='Phasenet')

plt.plot(window_sizes, ross_distances_p, label='Ross et al')

plt.yticks(np.arange(0, 250, step=15))

plt.ylabel('Mean absolute error (ms)')
plt.xlabel('Processed window size (s)')

plt.title('P phase picking results')

plt.legend()

# plt.show()

plt.savefig("results_p.pdf", bbox_inches='tight')


plt.cla()
plt.clf()


plt.plot(window_sizes, cospy_distances_s, label='Cospy')

plt.plot(window_sizes, phasenet_distances_s, label='Phasenet')

plt.plot(window_sizes, ross_distances_s, label='Ross et al')

plt.yticks(np.arange(0, 250, step=15))

plt.ylabel('Mean absolute error (ms)')
plt.xlabel('Processed window size (s)')

plt.title('S phase picking results')


plt.legend()


plt.savefig("results_s.pdf", bbox_inches='tight')


