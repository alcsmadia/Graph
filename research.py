import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('test.csv', names=['num1', 'num2'])

plt.plot(range(0,10), data['num2'], marker="o")
plt.title("Graph Title")
plt.show()