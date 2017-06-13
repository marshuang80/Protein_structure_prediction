import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


# Scatter plot of length vs accuracy
plt.plot(test_length,test_acc,'o')
plt.xlabel("Sequence length")
plt.ylabel("% accuracy)
z = np.ployfit(test_length,test_acc,1)
p = np.ploy1d(z)
plt.plot(test_length, p(test_length),'r--')

# Look for statistical significance
results = sm.OLS(test_acc,sm.add_constant(test_length)).fit()
print(results.summary())


# Bar graph for categories
struc = df["Secondary Structure"].values
test_struc = struc[5600:]
cat_acc = [(a,b) for a,b in zip(test_acc,test_struc)]

other = np.array([a[0] for a in cat_acc if a[1] == 'other'])
sheet = np.array([a[0] for a in cat_acc if a[1] == 'sheet'])
helix = np.array([a[0] for a in cat_acc if a[1] == 'helix'])

other_mean = other.mean()
helix_mean = helix.mean()
sheet_mean = sheet.mean()

other_std = other.std()
sheet_std = sheet.std()
helix_std = helix.std()

plt.bar(np.arange(3),(other_mean,sheet_mean,helix_mean),0.35,yerr=(other_std,sheet_std,helix_std)
,color = 'rgb')

plt.xticks(np.arange(3),
          ("other \n %f"%other_mean,
            "helix \n %f"%helix_mean,
            "sheet \n %f"%sheet_mean),
            fontsize=18)
plt.xlabel("Categories",fontsize=18)
plt.ylabel("Accuracy",fontsize=18)


# Categories scatter plot
helix_s, = plt.plot(np.arange(helix.shape[0]),helix,'bx')
sheet_s, = plt.plot(np.arange(sheet.shape[0])*3.7,sheet,'ro')
other_s, = plt.plot(np.arange(other.shape[0])*1.55,other,'g^')
plt.ylabel("% accuracy",fontsize=18)
plt.legend((helix_s,other_s,sheet_s),('helix','other','sheet'),loc = 'lower right', fontsize=10)


# ANOVA test

f_val, p_val = stats.f_oneway(sheet,helix,other)
if p_val < 0.0001:
    print("Statistically different based on ANOVA variance test")
else:
    print("Not statistically different")


