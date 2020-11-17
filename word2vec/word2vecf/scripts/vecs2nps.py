import numpy as np
import sys

fh = open(sys.argv[1],'r')
#fh=file(sys.argv[1])
foutname=sys.argv[2]
first=fh.readline()
size=first.strip().split()

wvecs=np.zeros((int(size[0]), int(size[1])),float)

vocab=[]
#print(fh.readlines())
for i,line in enumerate(fh):
    line = line.strip().split()
    #print(line)
    vocab.append(line[0])
    wvecs[i,] = np.array(line[1:]).astype(float)

np.save(foutname+".npy",wvecs)
