import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data  = pd.read_csv("overall_data.csv")
cols = data.columns
print(cols)
plt.semilogy(data[cols[0]], data[cols[1]],label=cols[1] )
plt.semilogy(data[cols[2]], data[cols[3]], label=cols[3] )
plt.semilogy(data[cols[4]], data[cols[5]], label=cols[5])
plt.semilogy(data[cols[6]], data[cols[7]], label=cols[7] )
plt.semilogy(data[cols[8]], data[cols[9]], label=cols[9] )
plt.semilogy(data[cols[10]], data[cols[11]], label=cols[11] )
plt.semilogy(data[cols[12]], data[cols[13]], label=cols[13] )

plt.legend()


plt.show()
