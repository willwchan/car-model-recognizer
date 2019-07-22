from glob2 import glob
import pandas as pd

df = pd.DataFrame(columns=['file','make'])

for image in glob('train/**/*.jpg'):
    dir_ = image.split('/')
    file_, make = dir_[-1], dir_[-2]

    df = df.append({
        'file': file_,
        'make': make
        }, ignore_index=True)

df.to_csv('labels.csv', index=False)
