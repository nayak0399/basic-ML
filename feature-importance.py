import matplotlib.pyplot as plt

x = ['Group1', 'Covalent-Radius(pm)1', 'Atomic-Number1', 'Atomic-Mass(a.u)1', 'Period1', 'Electro-Negativity1', 'Ionization-Energy(kJmol-1)1', 'Enthalpy-of-Fusion(kJmol-1)1', 'Density(gcm-3)1','Homo-Level', 'Lumo_Level']
x=['G', 'R', 'AN', 'AM','P', 'EN', 'IE', '$\Delta_{fus}$H' ,'$\\rho$','HOMO', 'LUMO']
feature_imp = [0.08913735, 0.02508969, 0.00938435, 0.10325313, 0.06042418, 0.03551675, 0.00258749, 0.0802988,0.00781287,0.02460417,0.07522234]
#variance = [1, 2, 7, 4, 2, 3]

#x_pos = [i for i, _ in enumerate(x)]
fig, ax = plt.subplots(figsize=(5,3), dpi=300)
[i.set_linewidth(1.0) for i in ax.spines.itervalues()]
ax.barh(x, feature_imp, color='green',alpha=0.8)
ax.set_ylabel("Features", fontsize=13)
ax.set_xlabel("Feature Importance",fontsize=13)
ax.set_title("Adsorbate Features",fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.set_yticks(x, x)
plt.tight_layout()
plt.savefig('Adsorbate_Features.tif')
plt.show()

