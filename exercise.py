import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('winequality-red.csv')
ind = dataset.iloc[:, 0:-1].values
dep = dataset.iloc[:, -1].values

decisionTreeRegressor = DecisionTreeRegressor (random_state = 0)
decisionTreeRegressor.fit(ind, dep)

# Quality vs Alcohol
plt.scatter(ind[:, -2], decisionTreeRegressor.predict(ind),
color="red")
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol vs Quality')
plt.show()

# Density vs Quality vs Alcohol
fig = plt.figure()
subplot = fig.add_subplot(111, projection='3d')
subplot.scatter(ind[:, -2], ind[:, -5],
decisionTreeRegressor.predict(ind), color="red")
plt.show()



# Construa uma árvore de decisão para regressão. Considere que quality" é a
# variável dependente. Exiba um gráco de duas dimensões de alcohol versus
# quality. Exiba um gráco de três dimensões de alcohol versus density versus
# quality.
