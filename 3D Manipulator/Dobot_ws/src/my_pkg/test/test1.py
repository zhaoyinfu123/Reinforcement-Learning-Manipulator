import numpy as np

a = np.arange(12).reshape(3, 4)
b = np.arange(12, 24).reshape(3, 4)
c = np.arange(24, 36).reshape(3, 4)
x = [a, b, c]
print(len(x))
d = np.arange(36, 48).reshape(3, 4)

z = 0
for i in range(len(x)):
    # print((d == x[i]).all())
    print(x[i])
    if (d == x[i]).all():
        z += 0
    else:
        z += 1
if z == len(x):
    print('no same')
else:
    print('have same')