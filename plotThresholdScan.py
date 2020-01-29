import numpy as np
import matplotlib.pyplot as plt

rawdata=np.loadtxt(fname='/data/deasy/DylanHsu/thresholdScan_training-scanSizeV1mm3-bs1000-dr0p50-48mm.txt')
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
det_bins = np.append(det_thresholds,[1.])
seg_bins = np.append(seg_thresholds,[1.])
#print(det_bins.size,seg_bins.size)
#print(dice_scores.shape)
#det_bins = det_thresholds+[1]
f=plt.figure()

mesh=plt.pcolormesh(det_bins,seg_bins,np.transpose(dice_scores,(1,0)),cmap='magma',vmin=0.5, vmax=np.amax(dice_scores))
colorbar=f.colorbar(mesh)
colorbar.set_label('Average Patient Dice score')
ax = f.add_subplot(111)
f.subplots_adjust(top=0.85)
ax.set_title('Threshold scan. 48mm network, training dataset')
ax.set_xlabel('Detection threshold')
ax.set_ylabel('Segmentation threshold')
plt.savefig('/data/deasy/DylanHsu/plots/thresholdScan.png')
plt.close(f)
