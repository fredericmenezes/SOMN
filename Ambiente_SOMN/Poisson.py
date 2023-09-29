import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

#atraso = poisson.rvs(mu=2, size=100, random_state=0)
atraso = [poisson.rvs(mu=4) for _ in range(1000)]
sns.histplot(atraso, discrete=True)
plt.xlim([-1,15])
plt.xlabel('Atraso (k)')
plt.ylabel('P(X=k)')
#atraso