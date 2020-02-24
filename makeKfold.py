import os
import random
from glob import glob

k=5

folder='/data/deasy/DylanHsu/N401_unstripped'
# this needs to have subfolder 'nifti', and files train.txt/test.txt
augcache='/data/deasy/DylanHsu/N401_unstripped/augcache'


f=open(os.path.join(folder,'train.txt'),'r')

cases=[]
for case in f:
 cases+=[case.rstrip()]
random.shuffle(cases)

subgroup_size=len(cases)//k
print('Found %d cases, which will be split into %d subgroups of %d' % (len(cases), k, subgroup_size))

subgroup_cases={}
for i in range(k):
 if i < k-1:
  subgroup_cases[i]=cases[i*subgroup_size:(i+1)*subgroup_size]
 else:
  subgroup_cases[i]=cases[(k-1)*subgroup_size:]
 print("Subgroup %d has %d cases" % (i+1,len(subgroup_cases[i])))
 sgf=open(os.path.join(folder,'subgroup%d.txt'%(i+1)),'w+')
 for case in subgroup_cases[i]:
  sgf.write("%s\n"%case)

for i in range(k):
 os.mkdir(os.path.join(folder,'subgroup%d'%(i+1)))
 train_folder=os.path.join(folder,'subgroup%d'%(i+1),'training')
 test_folder=os.path.join(folder,'subgroup%d'%(i+1),'testing')
 os.mkdir(train_folder)
 os.mkdir(test_folder)

 # training cases
 train_cases=[]
 for j in range(k):
  if i is j:
   continue
  train_cases+=subgroup_cases[j]
 test_cases=subgroup_cases[i]
 
 for case in train_cases:
  augpaths=glob(os.path.join(augcache,case)+'_*')
  for augpath in augpaths:
   os.symlink(augpath, os.path.join(train_folder, os.path.basename(augpath)))
 for case in test_cases:
  os.symlink(os.path.join(folder,'nifti',case), os.path.join(test_folder,case))
