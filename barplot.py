import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups = 8

means_atom = (0.865, 0.865, 0.865, 0.864, 0.352, 0.50, 0.4365, 0.498)
std_atom = (0.111, 0.111, 0.111, 0.111, 0.065, 0.27, 0.134, 0.130 )

means_molecule = (0.663, 0.651, 0.663, 0.663, 0.268, 0.286, 0.325, 0.3857)
std_molecule = (0.108, 0.105, 0.108, 0.108, 0.049, 0.13, 0.093, 0.09)


fig, ax = plt.subplots(figsize=(5,4),dpi=200)

index = np.arange(n_groups)
bar_width = 0.35

#opacity = 0.4
#error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, means_atom, bar_width,
                alpha=0.65, color='b',
                yerr=std_atom, 
               # error_kw=error_config,
                label='Atomic')

rects2 = ax.bar(index + bar_width, means_molecule, bar_width,
                alpha=0.65, color='g',
                yerr=std_molecule, 
                #error_kw=error_config,
                label='O-X')
                

[i.set_linewidth(1.0) for i in ax.spines.itervalues()]                
#ax.spines[axis].set_linewidth(0.5)
#ax.set_xlabel('Group')
ax.set_ylabel('Test RMSE (eV)', fontsize=12)

#ax.set_title('Scores by group and gender')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('OLS', 'PLS', 'Ridge', 'Lasso', 'Kernel \n Ridge','GPR', 'GBR', 'RFR'),rotation=60)
ax.set_xlabel('Regression Methods', fontsize=12)
ax.legend()
fig.tight_layout()
plt.savefig('model-error.jpg')
plt.show()

