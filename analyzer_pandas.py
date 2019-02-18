import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

import time

#### FIT simulated microlensing event

"""
merged = pd.read_pickle("merged_lc_ampli")
seed_dict = merged.groupby("id_E").groups
# print(seed_dict)

c1 = merged.id_E == "lm0322n10026"
temp = merged
tempE = temp.dropna(subset=["blue_E", "red_E"])
tempE = tempE[(tempE.blue_E<30) & (tempE.red_E<30)]
tempM = temp.dropna(subset=["blue_M", "red_M"])
tempM = tempM[(tempM.blue_M!=-99.) & (abs(tempM.blueerr_M)<10.) & (tempM.red_M!=-99.)]


arr1 = moving_window_smoothing(tempE.time.values, tempE.ampli_blue_E.values, 5)
arr2 = moving_window_smoothing(tempE.time.values, tempE.ampli_red_E.values, 5)
print(np.corrcoef(arr1[1], arr2[1]))


popt, pcov = curve_fit(microlensing_amplification, arr1[0], arr1[1], [0.5, 50000, 500])
flat_popt, flat_pcov = curve_fit(flat_curve, arr1[0], arr1[1], [-7])
print(popt, np.sqrt(np.diag(pcov)))
true_params = generate_microlensing_parameters(seed_dict["lm0322n10026"][0])
print(true_params)
plt.scatter(tempE.time.values, tempE.ampli_blue_E.values)
plt.plot(arr1[0], microlensing_amplification(arr1[0], *popt))
plt.plot(arr1[0], tempE.blue_E.mean()+2.5*np.log10(microlensing_amplification(arr1[0], *true_params)))
plt.show()"""


merged = pd.read_pickle('merged_lc_ampli')

### VISUALIZE INTERESTING STAR
temp = merged[merged['id_E']=='lm0322n26850']
plt.scatter(temp.time, temp.blue_E, color='green', marker='+')
plt.scatter(temp.time, temp.red_E, color='black', marker='+')
ax = plt.gca().twinx()
ax.scatter(temp.time, temp.blue_M, color='blue', marker='+')
ax.scatter(temp.time, temp.red_M, color='red', marker='+')
plt.show()

plt.subplot(2,1,1)
plt.scatter(temp.red_E, temp.blue_E, color='black', marker='+')
plt.gca().axis('equal')
plt.subplot(2,1,2)
plt.scatter(temp.red_M, temp.blue_M, color='pink', marker='+')
plt.gca().axis('equal')
plt.show()





### INTERACTIVE VIEW PERIODIC
from matplotlib.widgets import Slider
fig, ax = plt.subplots(1, 2, sharex=True)
plt.subplots_adjust(bottom=0.25)
sax = plt.axes([0.25, 0.1, 0.65, 0.03])
slide = Slider(sax, 'period', 0.1, 400, valinit=0.438174)
s1, = ax[0].plot(temp.time, temp.blue_E, '+')
s2, = ax[0].plot(temp.time, temp.red_E, '+')
s3, = ax[1].plot(temp.time, temp.blue_M, '+')
s4, = ax[1].plot(temp.time, temp.red_M, '+')


def update(period):
	new_time = temp.time%period
	s1.set_xdata(new_time)
	s2.set_xdata(new_time)
	s3.set_xdata(new_time)
	s4.set_xdata(new_time)

	ax[0].set_xlim([new_time.min()*0.9, new_time.max()*1.1])
	ax[1].set_xlim([new_time.min()*0.9, new_time.max()*1.1])
	fig.canvas.draw()

slide.on_changed(update)
plt.show()




WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"

corr_matrix = pd.read_pickle(WORKING_DIR_PATH+"corr_matrix").unstack()

c1 = corr_matrix[('smooth_red_E', 'smooth_blue_E')]>0.7
c2 = corr_matrix[('smooth_red_M', 'smooth_blue_M')]>0.7

#corr_matrix = corr_matrix[c1 | c2]
print(corr_matrix.dropna(axis=0, how='any', subset=(('smooth_red_E', 'smooth_blue_E'), ('smooth_red_M', 'smooth_blue_M')), inplace=True))
print(corr_matrix)

sns.kdeplot(corr_matrix[('smooth_red_E','smooth_blue_E')], corr_matrix[('smooth_red_M', 'smooth_blue_M')], palette="Greens_r", label='Data', legend=False)
sns.kdeplot(corr_matrix[('smooth_ampli_red_E','smooth_ampli_blue_E')], corr_matrix[('smooth_ampli_red_M', 'smooth_ampli_blue_M')], palette="Reds_r", label='Simulated', legend=False)
plt.xlabel("Correlation EROS")
plt.ylabel("Correlation MACHO")
cmap2 = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True, rot=-.4)
sns.jointplot(corr_matrix[('smooth_ampli_red_E', 'smooth_ampli_blue_E')], corr_matrix[('smooth_ampli_red_M', 'smooth_ampli_blue_M')])
sns.jointplot(corr_matrix[('smooth_red_E', 'smooth_blue_E')], corr_matrix[('smooth_red_M', 'smooth_blue_M')])
plt.show()


BINS = 100
plt.subplot(3,1,1)
plt.hist(corr_matrix[('smooth_red_E', 'smooth_blue_E')], color='gray', bins=BINS)
plt.hist(corr_matrix[('smooth_ampli_red_E', 'smooth_ampli_blue_E')], color='black', histtype='step', bins=BINS)
plt.title("EROS")
plt.subplot(3,1,2)
plt.hist(corr_matrix[('smooth_red_M', 'smooth_blue_M')], color='gray', bins=BINS)
plt.hist(corr_matrix[('smooth_ampli_red_M', 'smooth_ampli_blue_M')], color='black', histtype='step', bins=BINS)
plt.title("MACHO")
plt.subplot(3,1,3)
plt.hist(np.sqrt(corr_matrix[('smooth_red_M', 'smooth_blue_M')].pow(2)+corr_matrix[('smooth_red_E', 'smooth_blue_E')].pow(2)), color='gray', bins=BINS)
plt.hist(np.sqrt(corr_matrix[('smooth_ampli_red_M', 'smooth_ampli_blue_M')].pow(2)+corr_matrix[('smooth_ampli_red_E', 'smooth_ampli_blue_E')].pow(2)), color='black', bins=BINS, histtype='step')
plt.show()