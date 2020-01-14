import os
import sys

# sample data on names.txt

if len(sys.argv) < 3:
    print("Usage: python dataCleaner.py PICS_DIR DEST_DIR [candidata pngs file name]")
    exit(0)

ROOT_DIR = sys.argv[1]
SAMPLE_DIR = sys.argv[2]
IMG_LIST_DIR = 'names.txt'

if len(sys.argv)==4:
    IMG_LIST_DIR = sys.argv[3]

if not os.path.exists(SAMPLE_DIR):
    os.system('mkdir '+SAMPLE_DIR)

pngs = []
with open(IMG_LIST_DIR) as f:
    pngs = f.readlines()

for png in pngs:
    src = os.path.join(ROOT_DIR,png)
    dst = SAMPLE_DIR+'/'
    cmd_cp = 'cp {0} {1}'.format(src,dst)
    
    try:
        if os.system(cmd_cp) !=0:
            print(png)
            print(cmd_cp)
    except Exception as e:
        print(png)
        print(cmd_cp)
        print(e)
        exit(0)
    
