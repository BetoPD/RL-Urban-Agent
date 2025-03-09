import numpy as np
import matplotlib.pyplot as plt

city = [[2, 2, 1, 1, 2],
        [1, 2, 2, 1, 1],
        [2, 2, 1, 1, 1],
        [2, 1, 2, 2, 1],
        [1, 1, 1, 2, 1]]

city = np.array(city)

plt.imshow(city, cmap='viridis', interpolation='nearest')

# Street = 1
# House = 2
# Add a legend that explains the colors
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Street'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='House')],
           loc='upper left')


plt.show()
