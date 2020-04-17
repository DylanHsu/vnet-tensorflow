import os, sys
import datetime
import SimpleITK as sitk
from math import sqrt
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import random
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='Scan the optimal probability thresholds for detection & segmentation')

parser.add_argument('suffix'       , type=str  , help="suffix of the written label files")
parser.add_argument('data_dir'     , type=str  , help="location in which the training/testing folders reside")
parser.add_argument('threshold'    , type=float, help="threshold for detection")
parser.add_argument('seg_threshold', type=float, help="threshold for segmentation")
parser.add_argument("--write_labels", help="actually write the label Nifti files", action="store_true")
parser.add_argument('--dilate_threshold', type=float, help="threshold for dilating with gaussian kernel",default=0.0)
parser.add_argument("--overlay_freesurfer", help="actually write the label Nifti files", action="store_true") # use FS' mri_convert to make aseg.nii.gz files
args = parser.parse_args()

#image_name = 'img.nii.gz'
label_name = 'label_smoothed.nii.gz'
prob_name = 'probability_vnet'+'-'+args.suffix+'.nii.gz'
#prob_name = 'probability_centernet-%s.nii.gz'%args.suffix
stats_dir='/data/deasy/DylanHsu/SRS_N401/stats/stats'+'-'+args.suffix
try:
  os.makedirs(stats_dir)
except FileExistsError:
  pass

# setup Freesurfer data structures
if args.overlay_freesurfer:
  from srsPre.fsOneHotEncoding import fsDicts
  fsLabelDict, fsNamedGroups, fsNumberedGroups = fsDicts()
  fs_aseg_name = 'aseg.nii.gz'
  fsGroupConfusionMatrix = {}
  fsLabelToGroupMapping = {}
  for group, indices in fsNumberedGroups.items():
    fsGroupConfusionMatrix[group] = [0,0,0,0] #tp,fp,tn,fn
  for index in fsLabelDict:
    foundIndex=False
    for group, indices in fsNumberedGroups.items():
      if index in indices:
        fsLabelToGroupMapping[index] = group
        foundIndex=True
    #if not foundIndex:
    #  fsLabelToGroupMapping[index] = 'Null'
    #  print('FS index %d "%s" mapped to Null group'%(index,fsLabelDict[index]))
      
        


#for dataset in ['training','testing']:
for dataset in ['testing']:
  probfiles = glob(os.path.join(args.data_dir, dataset, '*', prob_name))
  cases = [os.path.basename(os.path.dirname(probfile)) for probfile in probfiles]
  #cases = cases[:5]
  random.shuffle(cases) # maximum parallel I/O
  ncases = len(cases)
  assert ncases>0
  
  avgPatientDice=0
  patientDiceSumSq=0
  avgPatientSpecificity=0
  avgPatientSensitivity=0
  fpmSum=0
  fnmSum=0
  tpmSum=0
  fpmSumSq=0
  fnmSumSq=0
  tpmSumSq=0
  tpvTotal=0 
  fpvTotal=0 
  tnvTotal=0 
  fnvTotal=0 
  globalLesionStats=[]
  for case in cases:
    print('analyzing case',case)
    #image_path = os.path.join(args.data_dir, dataset, case, 'img.nii.gz')
    label_path = os.path.join(args.data_dir, dataset, case, label_name)
    prob_path  = os.path.join(args.data_dir, dataset, case, prob_name)
    
    # Instantiate filters
    aif             = sitk.AddImageFilter()
    btif            = sitk.BinaryThresholdImageFilter()
    castImageFilter = sitk.CastImageFilter()
    ccFilter        = sitk.ConnectedComponentImageFilter()
    dgif            = sitk.DiscreteGaussianImageFilter()
    eif             = sitk.EqualImageFilter()
    hdif            = sitk.HausdorffDistanceImageFilter()
    mapFilter       = sitk.LabelImageToLabelMapFilter()
    lmmif           = sitk.LabelMapMaskImageFilter()
    maskImageFilter = sitk.MaskImageFilter()
    overlapFilter   = sitk.LabelOverlapMeasuresImageFilter()
    #pasteFilter     = sitk.PasteImageFilter()
    reader          = sitk.ImageFileReader()
    statFilter      = sitk.StatisticsImageFilter()
    writer          = sitk.ImageFileWriter()
    xif             = sitk.MultiplyImageFilter()
    
    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    trueLabelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    trueLabelShapeFilter.SetComputeFeretDiameter(True)
    seededSegLabelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    
    #reader.SetFileName(image_path)
    #image = reader.Execute()
    
    reader.SetFileName(label_path)
    trueLabel = reader.Execute()
    castImageFilter.SetOutputPixelType( sitk.sitkInt16 )
    trueLabel = castImageFilter.Execute(trueLabel)
    
    reader.SetFileName(prob_path)
    softmax = reader.Execute()
    
    if args.overlay_freesurfer:
      aseg_path  = os.path.join(args.data_dir, dataset, case, fs_aseg_name)
      reader.SetFileName(aseg_path)
      aseg = reader.Execute()
      # This is suboptimal, but I couldn't figure out the correct settings to get PasteImageFilter working
      #aseg = sitk.Resample(aseg, trueLabel.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor, trueLabel.GetOrigin(), trueLabel.GetSpacing(), trueLabel.GetDirection(), 0, aseg.GetPixelID())
      #castImageFilter.SetOutputPixelType( aseg.GetPixelID() )
      #pasteFilter.Execute( castImageFilter.Execute(trueLabel), aseg)
      

    #image_np = sitk.GetArrayFromImage(image)
    #softmax_np = np.transpose(sitk.GetArrayFromImage(softmax), (2,1,0))
    trueLabel_np = np.transpose(sitk.GetArrayFromImage(trueLabel), (2,1,0))
  
    # True label computations for connected component analysis
    cubicMmPerVoxel = trueLabel.GetSpacing()[0] * trueLabel.GetSpacing()[1] * trueLabel.GetSpacing()[2];
    trueLabelCC = ccFilter.Execute(trueLabel)
    trueLabelShapeFilter.Execute(trueLabelCC)
    trueLabelMap=mapFilter.Execute(trueLabelCC,0)
      
    #label_np = np.where(softmax_np > float(args.threshold), 1,0)
  
    # Create SITK label mask for this threshold
    #label = sitk.GetImageFromArray(np.transpose(label_np, (2,1,0)))
    #label.SetSpacing(trueLabel.GetSpacing())
    #label.SetOrigin(trueLabel.GetOrigin())
    #label.SetDirection(trueLabel.GetDirection())
    label = btif.Execute(softmax, float(args.threshold), 1e6, 1, 0)
    castImageFilter.SetOutputPixelType( sitk.sitkInt16 ) # need to have same pixel type for overlapFilter
    label = castImageFilter.Execute(label)
    
    #################################
    # Connected component analysis  #
    #################################
    # compute true positives, false positives, etc. for this threshold and patient
    labelCC = ccFilter.Execute(label)
    labelShapeFilter.Execute(labelCC)
    labelMap=mapFilter.Execute(labelCC,0)
     
    # List of predicted lesions that could be a false positive
    nonartifacts=list(range(1,labelShapeFilter.GetNumberOfLabels()+1))
    
    # Ignorant ham-fisted post processing ideas go here
    # Commented out for now
    #for j in range(1,labelShapeFilter.GetNumberOfLabels()+1):
      #if (labelShapeFilter.GetNumberOfPixels(j) * cubicMmPerVoxel) < 14.14: 
      # Remove <3mm diameter predictions
      #  nonartifacts.remove(j)
    
    # Compute patient dice for this threshold and patient
    if args.seg_threshold == args.threshold:
      overlapFilter.Execute(label, trueLabel)
      patient_dice = overlapFilter.GetDiceCoefficient()
      seededSegLabel = label
    else:
      # Build a Segmentation mask by using the detections as seeds
      # and considering voxels connected to those seeds with a reasonable probability
      # to be the segmentation prediction.
      #segLabel_np = np.where(softmax_np > float(args.seg_threshold), 1,0)
  
      # Create SITK label mask for this threshold
      #segLabel = sitk.GetImageFromArray(np.transpose(segLabel_np, (2,1,0)))
      #segLabel.SetSpacing(trueLabel.GetSpacing())
      #segLabel.SetOrigin(trueLabel.GetOrigin())
      #segLabel.SetDirection(trueLabel.GetDirection())
      segLabel = btif.Execute(softmax, float(args.seg_threshold), 1e6, 1, 0)
      castImageFilter.SetOutputPixelType( sitk.sitkInt16 ) # need to have same pixel type for overlapFilter
      segLabel = castImageFilter.Execute(segLabel)
      
      segLabelCC = ccFilter.Execute(segLabel)
      segLabelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
      segLabelShapeFilter.Execute(segLabelCC)
      segLabelMap=mapFilter.Execute(segLabelCC,0)

      # Begin Double loop over the Segmentation mask and the Detection mask 
      # to pick out the lower-confidence Segmentation connected components which
      # have a high-confidence Detection seed.
      seededSegLabel = xif.Execute(label, 0.0)
      for i in range(1,segLabelShapeFilter.GetNumberOfLabels()+1):
        seededSegCCs=[]
        lmmif.SetLabel(i)
        segLesion = lmmif.Execute(segLabelMap, segLabel)
        for j in nonartifacts: # nonartifacts holds Detection seed indices
          lmmif.SetLabel(j)
          seedLesion = lmmif.Execute(labelMap, label)
          overlapFilter.Execute(segLesion, seedLesion)
          if overlapFilter.GetDiceCoefficient() > 0.0:
            seededSegCCs.append(i)
            seededSegLabel=aif.Execute(seededSegLabel, segLesion)
            break


      if args.dilate_threshold > 0.0:
        castImageFilter.SetOutputPixelType( sitk.sitkFloat32 )
        dgif.SetVariance(1)
        blurredSeededSegLabel = dgif.Execute(castImageFilter.Execute(seededSegLabel))
        seededSegLabel = btif.Execute(blurredSeededSegLabel, float(args.dilate_threshold), 1e6, 1, 0)
        castImageFilter.SetOutputPixelType( sitk.sitkInt16 )
        seededSegLabel = castImageFilter.Execute(seededSegLabel) 
      overlapFilter.Execute(seededSegLabel, trueLabel)
      patient_dice = overlapFilter.GetDiceCoefficient()

    avgPatientDice += patient_dice
    patientDiceSumSq += patient_dice*patient_dice
    

    ###########################
    # DGH TO DO
    # Do the Per-Lesion analysis based on the Seeded Segmentation when appropriate
    ###########################
    
    seededSegLabelCC = ccFilter.Execute(seededSegLabel)
    seededSegLabelShapeFilter.Execute(seededSegLabelCC)
    seededSegLabelMap=mapFilter.Execute(seededSegLabelCC,0)
    # redefine list of nonartifacts here
    # no ignorant postprocessing to remove artifacts for right now
    nonartifacts = list(range(1, seededSegLabelShapeFilter.GetNumberOfLabels()+1))
    falsePositivesList = nonartifacts.copy()
    
    # List of true lesions that could be a false negative
    falseNegativesList = list(range(1,trueLabelShapeFilter.GetNumberOfLabels()+1))
    
    # Double loop over the predicted and true lesions
    patientLesionStats=[]
    for i in range(1,trueLabelShapeFilter.GetNumberOfLabels()+1):
      overlappingPredictions=[]
      lmmif.SetLabel(i)
      true_lesion = lmmif.Execute(trueLabelMap, trueLabel)
      found_pred = False
      for j in nonartifacts:
        lmmif.SetLabel(j)
        predicted_lesion = lmmif.Execute(seededSegLabelMap, seededSegLabel)
        overlapFilter.Execute(predicted_lesion, true_lesion)
        if overlapFilter.GetDiceCoefficient() > 0.0:
          overlappingPredictions.append(j)
          # Mark this lesion as detected for the detection efficiency analysis
          found_pred = True
          if j in falsePositivesList:
            falsePositivesList.remove(j)
      if found_pred:
        if i in falseNegativesList:
          falseNegativesList.remove(i)
        if len(overlappingPredictions)>0:
          totalPrediction = None
          for j in overlappingPredictions:
            lmmif.SetLabel(j)
            if totalPrediction is None: 
              totalPrediction = lmmif.Execute(seededSegLabelMap, seededSegLabel)
            else:
              totalPrediction = aif.Execute(totalPrediction, lmmif.Execute(seededSegLabelMap, seededSegLabel))
          overlapFilter.Execute(totalPrediction, true_lesion)
          specificDice = overlapFilter.GetDiceCoefficient()
          hdif.Execute(true_lesion, totalPrediction)
          hausdorffDistance = hdif.GetHausdorffDistance()

        else:
          specificDice=0.
          hausdorffDistance=999.
      else:
        specificDice=0.
        hausdorffDistance=999.
      statFilter.Execute(true_lesion)
      volumetricDiameter = ((cubicMmPerVoxel * 6.*statFilter.GetSum()/3.14159) ** (1./3.))
      feretDiameter=trueLabelShapeFilter.GetFeretDiameter(i)
      lesionStatsDict={
        'case'               : case,
        'index'              : i,
        'feretDiameter'      : feretDiameter,
        'volumetricDiameter' : volumetricDiameter,
        'specificDice'       : specificDice,
        'hausdorffDistance'  : hausdorffDistance,
      }
      patientLesionStats.append(lesionStatsDict)
    globalLesionStats += patientLesionStats
    
    ###################
    # end CC analysis #
    ###################
  
    # Per lesion analysis
    fpmIncrement = len(falsePositivesList)
    fnmIncrement = len(falseNegativesList)
    tpmIncrement = len(nonartifacts) - len(falsePositivesList) 
    fpmSum += fpmIncrement
    fnmSum += fnmIncrement
    tpmSum += tpmIncrement
    fpmSumSq += fpmIncrement**2
    fnmSumSq += fnmIncrement**2
    tpmSumSq += tpmIncrement**2
  
    andOp = sitk.AndImageFilter()
    notOp = sitk.NotImageFilter()
    notTrueLabel = notOp.Execute(trueLabel)
    notSeededSegLabel = notOp.Execute(seededSegLabel)
  
    # Per voxel analysis
    truePositiveMask = andOp.Execute(trueLabel,seededSegLabel)
    statFilter.Execute(truePositiveMask)
    tpv = statFilter.GetSum()
    falsePositiveMask = andOp.Execute(notTrueLabel,seededSegLabel)
    statFilter.Execute(falsePositiveMask)
    fpv = statFilter.GetSum()
    trueNegativeMask = andOp.Execute(notTrueLabel,notSeededSegLabel)
    statFilter.Execute(trueNegativeMask)
    tnv = statFilter.GetSum()
    falseNegativeMask = andOp.Execute(trueLabel,notSeededSegLabel)
    statFilter.Execute(falseNegativeMask)
    fnv = statFilter.GetSum()
  
    patient_sensitivity = tpv/(tpv+fnv)
    patient_specificity = tnv/(tnv+fpv)
    avgPatientSensitivity += patient_sensitivity
    avgPatientSpecificity += patient_specificity
    tpvTotal += tpv
    fpvTotal += fpv
    tnvTotal += tnv
    fnvTotal += fnv

    if args.write_labels:
      seededSegLabelFileName = 'label-seed%.3f-seg%.3f-%s'%(args.threshold,args.seg_threshold,args.suffix)
      seededSegLabelFileName = seededSegLabelFileName.replace(".","p") + ".nii.gz"
      seededSegLabelPath = os.path.join(args.data_dir, dataset, case, seededSegLabelFileName)
      writer.SetFileName(seededSegLabelPath)
      castImageFilter.SetOutputPixelType( sitk.sitkInt16 )
      seededSegLabel=castImageFilter.Execute(seededSegLabel)
      writer.Execute(seededSegLabel)
      diceFileName = 'dice-seed%.3f-seg%.3f-%s'%(args.threshold,args.seg_threshold,args.suffix)
      diceFileName = diceFileName.replace(".","p") + ".txt"
      dicePath = os.path.join(args.data_dir,dataset,case,diceFileName)
      f_dice = open(dicePath,'w')
      f_dice.write("%s %.3f %d %d\n"%(case, patient_dice, len(falsePositivesList), len(falseNegativesList)))
      f_dice.close()
    if args.overlay_freesurfer:
      # apply the True Positive, etc. masks to the Freesurfer label image and
      # update the Freesurfer groups' confusion matrix accordingly
      confusionMasks = [truePositiveMask,falsePositiveMask,trueNegativeMask,falseNegativeMask]
      for c in range(len(confusionMasks)):
        maskedAseg = maskImageFilter.Execute(aseg, confusionMasks[c])
        #for fsLabel in fsLabelDict:
        for fsLabel in fsLabelToGroupMapping:
          # Check how many voxels in this masked image are equal to integer label fsLabel
          statFilter.Execute(eif.Execute( maskedAseg, fsLabel))
          labelVoxels = statFilter.GetSum()
          # Increment the accounting
          group = fsLabelToGroupMapping[fsLabel]
          fsGroupConfusionMatrix[group][c] += labelVoxels
          #print('adding %d to group=%s,c=%d'%(labelVoxels,group,c))
    #print(fsGroupConfusionMatrix['CerebralWM'])

  ncasesf=float(ncases)  
  avgPatientDice /= ncasesf
  patientFpm = fpmSum/ncasesf
  patientFnm = fnmSum/ncasesf
  patientTpm = tpmSum/ncasesf
  patientFpmError = sqrt( fpmSumSq/ncasesf - patientFpm**2)
  patientFnmError = sqrt( fnmSumSq/ncasesf - patientFnm**2)
  patientTpmError = sqrt( tpmSumSq/ncasesf - patientTpm**2)
  avgPatientDiceError = sqrt( patientDiceSumSq/ncasesf - avgPatientDice**2)
  avgPatientSensitivity /= ncasesf
  avgPatientSpecificity /= ncasesf
  detEff = tpmSum / float(tpmSum + fnmSum)
   
  # commence error propagation for the detection efficiency...
  if tpmSum is 0:
    tpmRelativeErrorSq=0
  else:
    tpmRelativeErrorSq = 1./float(tpmSum)
  if fnmSum is 0:
    fnmRelativeErrorSq=0
  else:
    fnmRelativeErrorSq = 1./float(fnmSum)
  detEffError = sqrt( tpmRelativeErrorSq + fnmRelativeErrorSq )*detEff/(detEff+1.)
  
  filename = 'lesions_%s-seed%.3f-seg%.3f-%s'%(dataset,args.threshold,args.seg_threshold,args.suffix)
  filename = filename.replace(".","p")
  filename += ".txt"
  f_lesions = open(os.path.join(stats_dir, filename),'w')
  avgLesionDice = 0.
  avgLesionHausdorff = 0.
  lesionDiceSumSq = 0.
  lesionHausdorffSumSq = 0.
  n_lesions=0
  for lesion in globalLesionStats:
    f_lesions.write("%s %d %.3f %.3f %.3f %.3f\n"%(lesion['case'],lesion['index'],lesion['feretDiameter'],lesion['volumetricDiameter'],lesion['specificDice'],lesion['hausdorffDistance']))
    if lesion['specificDice'] > 0:
      avgLesionDice += lesion['specificDice']
      avgLesionHausdorff += lesion['hausdorffDistance']
      lesionDiceSumSq += lesion['specificDice']**2
      lesionHausdorffSumSq += lesion['hausdorffDistance']**2
      n_lesions+=1
  f_lesions.close()
  
  if n_lesions > 0:
    avgLesionDice /= float(n_lesions)
    avgLesionHausdorff /= float(n_lesions)
    avgLesionDiceError = sqrt( lesionDiceSumSq/float(n_lesions) - avgLesionDice**2)
    avgLesionHausdorffError = sqrt( lesionHausdorffSumSq/float(n_lesions) - avgLesionHausdorff**2)
  else:
    avgLesionDice = 0.
    avgLesionDiceError = 0.
  
  filename = 'stats_%s-seed%.3f-seg%.3f-%s'%(dataset,args.threshold,args.seg_threshold,args.suffix)
  filename = filename.replace(".","p")
  filename += ".txt"
  f_stats = open(os.path.join(stats_dir, filename),'w')
  # header
  # threshold seg_threshold avgPatientDice avgPatientDiceError detEff detEffError avgLesionDice avgLesionDiceError avgLesionHausdorff, avgLesionHausdorffError patientFpm patientFpmError patientFnm patientFnmError patientTpm patientTpmError avgPatientSensitivity avgPatientSpecificity tpvTotal fpvTotal tnvTotal fnvTotal
  statsTuple = (args.threshold,args.seg_threshold, avgPatientDice,avgPatientDiceError,detEff,detEffError,avgLesionDice,avgLesionDiceError,avgLesionHausdorff,avgLesionHausdorffError,patientFpm,patientFpmError,patientFnm,patientFnmError,patientTpm,patientTpmError,avgPatientSensitivity,avgPatientSpecificity,tpvTotal,fpvTotal,tnvTotal,fnvTotal)
  f_stats.write("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.6f %.6f %d %d %d %d\n"%(statsTuple))
  f_stats.close()

  if args.overlay_freesurfer:
    filename = 'freesurfer_%s-seed%.3f-seg%.3f-%s'%(dataset,args.threshold,args.seg_threshold,args.suffix)
    filename = filename.replace(".","p")
    filename += ".txt"
    f_freesurfer = open(os.path.join(stats_dir, filename),'w')
    for group in fsGroupConfusionMatrix:
      matrix=fsGroupConfusionMatrix[group] 
      f_freesurfer.write("%s %d %d %d %d %d\n"%(group,matrix[0],matrix[1],matrix[2],matrix[3],matrix[0]+matrix[1]+matrix[2]+matrix[3]))
    f_freesurfer.close()


