import numpy as np

s1 = [0.8857, 0.8880, 0.8746, 0.8818, 0.8841]
s2 = [0.8512, 0.8482, 0.8425, 0.8413, 0.8531]

m1 = np.mean(s1)
std1 = np.std(s1)

m2 = np.mean(s2)
std2 = np.std(s2)

print(m1, m1 - std1, m1 + std1)
print(m2, m2 - std2, m2 + std2)
