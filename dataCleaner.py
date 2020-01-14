import os


# sample data on names.txt

ROOT_DIR = 'Data/'
SAMPLE_DIR = 'SampleData'
IMG_LIST_DIR = 'names.txt'


if not os.path.exists(SAMPLE_DIR):
    os.system('mkdir '+SAMPLE_DIR)

pngs = []
with open(IMG_LIST_DIR) as f:
    pngs = f.readlines()

for png in pngs:
    src = ROOT_DIR + png
    dst = SAMPLE_DIR+'/'
    cmd_cp = 'cp {0} {1}'.format(src,dst)
    if not os.system(cmd_cp):
        print(png)

