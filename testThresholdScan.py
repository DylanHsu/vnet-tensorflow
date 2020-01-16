import NiftiDataset
import os
import datetime
import SimpleITK as sitk
import math
import numpy as np
from glob import glob

# Info to spit out in a job
# threshold, patient dice, lesion size, false positives/patient, false negatives/patient, voxel confusion matrix, lesion detection efficiency, sensitivity, specificity

# 1 job per case not per threshold, to avoid extra file I/O

# parameters: case, evaluation suffix, data directory, 

# f_specificDice = open(os.path.join(FLAGS.data_dir,case,'specificDice%s.txt'%FLAGS.suffix), 'w')
# f_specificDice.write("%s %d %.2f %.4f %.4f %.4f %.3f\n"%(case, i, volumetric_d, intensity_balance, q10_balance, q90_balance, specific_dice))
# f_specificDice.close()


#hardcoded for now
data_dir = '/lila/data/deasy/DylanHsu/N200/training'
#probname = 'probability_vnet-noWeight.nii.gz'
probname = 'probability_vnet-4Layers-noWeight-dr0p2.nii.gz'
step = 0.01
start = 0.3
end = 1

probfiles = glob(os.path.join(data_dir, '*', probname))
cases = [os.path.dirname(probfile) for probfile in probfiles]
#cases = cases[:10]
ncases = len(cases)
print(ncases, "cases")

avg_patient_dice_dict={}
false_positives_dict={}
false_negatives_dict={}
true_positives_dict={}
true_negatives_dict={}
for threshold in np.arange(start,end+step,step):
  avg_patient_dice_dict[threshold] = 0
  false_positives_dict[threshold] = 0
  false_negatives_dict[threshold] = 0
  true_positives_dict[threshold] = 0
  true_negatives_dict[threshold] = 0
for case in cases:
  print('analyzing case',case)
  image_path = os.path.join(data_dir, case, 'img.nii.gz')
  label_path = os.path.join(data_dir, case, 'label_smoothed.nii.gz')
  prob_path  = os.path.join(data_dir, case, probname)
  
  reader = sitk.ImageFileReader()
  #reader.SetFileName(image_path)
  #image = reader.Execute()
  
  reader.SetFileName(label_path)
  true_label = reader.Execute()
  
  reader.SetFileName(prob_path)
  softmax = reader.Execute()
  
  #image_np = sitk.GetArrayFromImage(image)
  softmax_np = np.transpose(sitk.GetArrayFromImage(softmax), (2,1,0))
  true_label_np = np.transpose(sitk.GetArrayFromImage(true_label), (2,1,0))

  # Instantiate filters
  castImageFilter = sitk.CastImageFilter()
  ccFilter = sitk.ConnectedComponentImageFilter()
  mapFilter = sitk.LabelImageToLabelMapFilter()
  mif=sitk.LabelMapMaskImageFilter()
  overlapFilter = sitk.LabelOverlapMeasuresImageFilter()
  statFilter = sitk.StatisticsImageFilter()
  labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
  trueLabelShapeFilter = sitk.LabelShapeStatisticsImageFilter()

  # True label computations for connected component analysis
  # Don't need to re-perform these for every threshold!
  cubicMmPerVoxel = true_label.GetSpacing()[0] * true_label.GetSpacing()[1] * true_label.GetSpacing()[2];
  trueLabelCC = ccFilter.Execute(true_label)
  trueLabelShapeFilter.Execute(trueLabelCC)
  trueLabelMap=mapFilter.Execute(trueLabelCC,0)
  for threshold in np.arange(start,end+step,step):
    label_np = np.float32(softmax_np) - threshold + 0.500001
    label_np = np.clip(label_np, 0., 1.)
    label_np = np.rint(label_np)

    # Create SITK label mask for this threshold
    label = sitk.GetImageFromArray(np.transpose(label_np, (2,1,0)))
    label.SetSpacing(true_label.GetSpacing())
    label.SetOrigin(true_label.GetOrigin())
    label.SetDirection(true_label.GetDirection())
    castImageFilter.SetOutputPixelType( true_label.GetPixelID() ) # need to have same pixel type for overlapFilter
    label = castImageFilter.Execute(label)
    
    # Compute patient dice for this threshold and patient
    overlapFilter.Execute(label, true_label)
    patient_dice = overlapFilter.GetDiceCoefficient()
    avg_patient_dice_dict[threshold] += patient_dice

    # Connected component analysis:
    # compute true positives, false positives, etc. for this threshold and patient
    labelCC = ccFilter.Execute(label)
    labelShapeFilter.Execute(labelCC)
    labelMap=mapFilter.Execute(labelCC,0)
     
    # List of true lesions that could be a false negative
    falseNegativesList = list(range(1,trueLabelShapeFilter.GetNumberOfLabels()+1))
    
    # List of predicted lesions that could be a false positive
    nonartifacts=list(range(1,labelShapeFilter.GetNumberOfLabels()+1))
    for j in range(1,labelShapeFilter.GetNumberOfLabels()+1):
      # Remove <3mm diameter predictions
      if (labelShapeFilter.GetNumberOfPixels(j) * cubicMmPerVoxel) < 14.14: 
        nonartifacts.remove(j)
    falsePositivesList = nonartifacts.copy()
    
    # Double loop over the predicted and true lesions
    for i in range(1,trueLabelShapeFilter.GetNumberOfLabels()+1):
      mif.SetLabel(i)
      true_lesion = mif.Execute(trueLabelMap, true_label)
      found_pred = False
      specific_dice = 0.0
      for j in nonartifacts:
        mif.SetLabel(j)
        predicted_lesion = mif.Execute(labelMap, label)
        overlapFilter.Execute(predicted_lesion, true_lesion)
        if overlapFilter.GetDiceCoefficient() > 0.0 and not found_pred:
          found_pred = True
          specific_dice = overlapFilter.GetDiceCoefficient()
          if j in falsePositivesList:
            falsePositivesList.remove(j)
      if found_pred and i in falseNegativesList:
        falseNegativesList.remove(i)
      #statFilter.Execute(true_lesion)
      #volumetric_d = ((cubicMmPerVoxel * 6.*statFilter.GetSum()/3.14159) ** (1./3.))

    # end CC analysis
    false_positives_dict[threshold] += len(falsePositivesList)
    false_negatives_dict[threshold] += len(falseNegativesList)
    true_positives_dict[threshold] += len(nonartifacts) - len(falsePositivesList) 
    #true_negatives_dict[threshold] = trueLabelShapeFilter.GetNumberOfLabels()
print("{0:>8s} {1:>8s} {2:>8s} {3:>8s} {4:>8s}".format("Thresh.","P.Dice","FP/p","FN/p","Sd."))
for threshold in np.arange(start,end+step,step):
  #avg_dice_dict[threshold] = 0
  tp = true_positives_dict[threshold]
  fn = false_negatives_dict[threshold]
  fp = false_positives_dict[threshold]
  sensitivity = tp/float(tp+fn)
  print('{0:>8.3f} {1:>8.3f} {2:>8.2f} {3:>8.2f} {4:>8.3f}'.format(threshold,avg_patient_dice_dict[threshold]/ncases,false_positives_dict[threshold]/float(ncases),false_negatives_dict[threshold]/float(ncases),sensitivity))
