import os
import shutil
import pandas as pd

#create classes in both train and valid folders
names = pd.read_csv('names.csv')

for i in range(196):
  os.makedirs('train/'+names.iloc[i,0])
  os.makedirs('val/'+names.iloc[i,0])

labels = pd.read_csv('labels.csv')
labels = dict(labels.to_numpy())

for img in os.listdir('train_no_classes'):
  dest = labels[img]
  shutil.move('train_no_classes/'+img,'train/'+dest)

for img in os.listdir('val_no_classes'):
  dest = labels[img]
  shutil.move('val_no_classes/'+img,'val/'+dest)

print('DONE')