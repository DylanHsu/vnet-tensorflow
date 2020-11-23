import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

vmin=0.5

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
  new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
    cmap(np.linspace(minval, maxval, n)))
  return new_cmap
#rawdata=np.loadtxt(fname='/data/deasy/DylanHsu/thresholdScan_training-scanSizeV1mm3-bs1000-dr0p50-48mm.txt')
rawdata=np.loadtxt(fname='/data/deasy/DylanHsu/SRS_N514/stats/stats-mrct-gammaSched-subgroupX.txt')
det_thresholds=np.unique(rawdata[:,0])
seg_thresholds=np.unique(rawdata[:,1])
#print(det_thresholds,seg_thresholds)
dice_scores = np.zeros((det_thresholds.size,seg_thresholds.size))
a=0
for iD in range(det_thresholds.size):
 for iS in range(seg_thresholds.size):
  for iR in range(rawdata.shape[0]):
   if not (rawdata[iR,0]==det_thresholds[iD] and rawdata[iR,1]==seg_thresholds[iS]):
    continue
   dice_scores[iD,iS] = rawdata[iR,2]
   a+=1
#print(dice_scores)
vmax=np.amax(dice_scores)
det_bins = np.append(det_thresholds,[1.])
seg_bins = np.append(seg_thresholds,[1.])
#print(det_bins.size,seg_bins.size)
#print(dice_scores.shape)
#det_bins = det_thresholds+[1]
f=plt.figure()

#cmap = plt.get_cmap('bone_r')
#cmap = plt.get_cmap('afmhot_r')
#cmap = plt.get_cmap('pink_r')
cmap = plt.get_cmap('GnBu')
#new_cmap = truncate_colormap(cmap, 0, 0.9)
new_cmap = truncate_colormap(cmap, 0, 1.0)
mesh=plt.pcolormesh(det_bins,seg_bins,np.transpose(dice_scores,(1,0)),cmap=new_cmap,vmin=vmin, vmax=vmax)
colorbar=f.colorbar(mesh)
colorbar.set_label('Average patient Dice score')
ax = f.add_subplot(111)
#f.subplots_adjust(top=0.85)
#ax.set_title('Threshold scan. 48mm network, training dataset')
f.subplots_adjust(top=0.95)
ax.set_title('')
ax.set_xlabel('Detection threshold')
ax.set_ylabel('Segmentation threshold')
plt.savefig('/data/deasy/DylanHsu/plots/thresholdScan.png')
plt.close(f)

det_bins_surf = np.zeros((det_thresholds.size))
for i in range(det_thresholds.size):
  det_bins_surf[i] = (det_bins[i]+det_bins[i+1])/2.
seg_bins_surf = np.zeros((seg_thresholds.size))
for i in range(seg_thresholds.size):
  seg_bins_surf[i] = (seg_bins[i]+seg_bins[i+1])/2.

f=plt.figure()
ax = f.add_subplot(111,projection='3d')
X, Y = np.meshgrid(det_bins_surf,seg_bins_surf)
dice_scores[dice_scores<vmin]=vmin#np.nan
#cmap = plt.get_cmap('coolwarm')
#new_cmap = truncate_colormap(cmap, vmin,vmax)
ax.plot_surface(X,Y,np.transpose(dice_scores,(1,0)), cmap='coolwarm', edgecolor='none',vmin=vmin,vmax=vmax)

ax.set_title('')
ax.set_zlim(vmin,vmax)
ax.set_xlabel('Detection threshold')
ax.set_ylabel('Segmentation threshold')
ax.azim=135
plt.savefig('/data/deasy/DylanHsu/plots/thresholdScanSurf.png')
plt.close(f)




