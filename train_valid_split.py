import os
import shutil
import random

os.makedirs('val_no_classes')

num_img = len(os.listdir('train_no_classes'))
val_size = (int)(num_img * 0.2)

count = 0

for img in os.listdir('train_no_classes'):
    if random.randint(0,1) < 1 and count < val_size:
        shutil.move('train_no_classes/'+img, 'val_no_classes')
        count = count + 1

print('DONE')
