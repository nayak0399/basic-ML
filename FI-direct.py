import matplotlib.pyplot as plt
import csv
import pandas as pd
df = pd.read_csv('Feature_vs_coeff.csv')
y=df['coefficient']
x=df['features name']
#print df

fig, ax = plt.subplots(figsize=(5,3), dpi=300)
[i.set_linewidth(1.0) for i in ax.spines.itervalues()]
ax.barh(x, y, color='green',alpha=0.8)
ax.set_ylabel("Features", fontsize=13)
ax.set_xlabel("Importance",fontsize=13)
ax.set_title(" ",fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=10)
#ax.set_yticks()
#ax.set_xticks()
#ax.set_yticklabels(())
plt.tight_layout()
plt.savefig('Features.tif')
plt.show()

