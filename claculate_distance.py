import numpy as np
import math

a = [2.7663074, -2.3617246,  0.0816548]
b = [3.0334077,  2.6352427,  0.03353234]

v = np.subtract(a, b)
print(v)
v_2 = np.power(v,2)
print(v_2)
v = v_2.tolist()
dis_v = sum(v)
print(dis_v)
dis_v = math.sqrt(dis_v)
print(dis_v)


