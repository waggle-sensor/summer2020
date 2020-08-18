import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

od = OrderedDict()
od['13:52:56'] = 0.0
od['13:53:02'] = 1.0
od['13:53:07'] = 0.5

tuple_list = list(od.items())
x, y = zip(*tuple_list)

xx = np.array(x)
yy = np.array(y)

print(xx)
print(yy)

plt.plot(xx, yy, marker='o')
plt.show()
