import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

data = pd.read_csv("cost_revenue_clean.csv")
# data.info()
# data.describe()

X = DataFrame(data, columns = ['production_budget_usd'])
y = DataFrame(data ,columns= ['worldwide_gross_usd'])

#plt.figure(figsize=(10,6))
#plt.scatter(X, y, alpha=0.3)
#plt.title('Film Cost vs Global Revenue')
#plt.xlabel('Production Budget $')
#plt.ylabel('Worldwide Gross $')
#plt.ylim(0, 3000000000)
#plt.xlim(0, 450000000)
#plt.show()

lr = LinearRegression()
lr.fit(X, y)

slope_coefficient = lr.coef_ # theta1 slope
intercept_ = lr.intercept_   # 

plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.3)
plt.plot(X, lr.predict(X) , color = "red", linewidth=2)

plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()
r2 = lr.score(X, y)
print("R2 score:",end=" ")
print(r2)
