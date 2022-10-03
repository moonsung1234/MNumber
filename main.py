
import matplotlib.pyplot as plt
from util import NCreater

nc = NCreater()

# run
for _ in range(5) :
    result = nc.create(123456789)

    plt.imshow(result)
    plt.show()