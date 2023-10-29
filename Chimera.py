#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 00:02:53 2020

@author: isaac
"""

from PIL import Image
import numpy as np
from skimage.measure import compare_ssim
import time
import os

#initialization
startTime = time.time()
img = Image.open('huugecat.jpg').convert('L')
img.show()
ary = np.array(img)
ary = ary[0:(img.size[1]-img.size[1]%4),0:(img.size[0]-img.size[0]%4)]
ary = ary.transpose()
block_arr = []
#img.save(os.path.join('./imagejpegcomp.jpeg'),'JPEG',quality = 50)
#divide image into 4x4 blocks
for i in range(ary.shape[1]//4):
    for j in range(ary.shape[0]//4):
        block = np.zeros((4,4))
        for k in range(4):
            for l in range(4):
                block[l,k] = ary[j*4+l,i*4+k]
        block_arr.append(block)
coefA,coefB,coefC =[],[],[]
T = []
#Normalize blocks
for i in range(ary.shape[1]//4):
    for j in range(ary.shape[0]//4):
        coefA.append(np.min(block_arr[(ary.shape[0]//4)*i+j]))
        x_min = np.min(block_arr[(ary.shape[0]//4)*i+j])
        x_max = np.max(block_arr[(ary.shape[0]//4)*i+j])
        coefB.append(x_max - x_min)
        x_i = block_arr[(ary.shape[0]//4)*i+j]
        if x_max != x_min:
            block = ((x_i-x_min)/(x_max-x_min))*24
        else:
            block = np.ones([4,4])*24
        T.append(block)
#create mask library
vectors = []
vectors.append(np.array([[24,24,24,24]]))
vectors.append(np.array([[0,8,16,24]]))
vectors.append(np.array([[24,26,8,0]]))
vectors.append(np.array([[0,24,24,24]]))
vectors.append(np.array([[24,24,24,0]]))
vectors.append(np.array([[24,24,0,0]]))
vectors.append(np.array([[0,24,24,0]]))
vectors.append(np.array([[0,0,24,24]]))
vectors.append(np.array([[24,0,0,24]]))
vectors.append(np.array([[24,0,0,0]]))
vectors.append(np.array([[0,24,0,0]]))
vectors.append(np.array([[0,0,24,0]]))
vectors.append(np.array([[0,0,0,24]]))
vectors.append(np.array([[12,12,24,24]]))
vectors.append(np.array([[12,24,24,12]]))
vectors.append(np.array([[24,24,12,12]]))
mskLib = []
for i in range(16):
    for j in range(16):
        mskLib.append((vectors[i].transpose().dot(vectors[j]))/24)
mskLib[15]=np.array([[0,8,16,24],[8,0,8,16],[16,8,0,8],[24,16,8,0]])
mskLib[221] = np.rot90(mskLib[15],2)
mskLib[222] = np.array([[24,16,8,0],[16,24,16,8],[8,16,24,8],[0,8,16,24]])
mskLib[223] = np.rot90(mskLib[222],2)
mskLib[237] = np.array([[8,8,8,0],[8,24,16,8],[8,16,24,8],[0,8,8,8]])
mskLib[238] = np.rot90(mskLib[237],2)
mskLib[239] = np.array([[0,12,0,0],[12,24,0,0],[12,24,0,0],[0,12,0,0]])
mskLib[253] = np.rot90(mskLib[239])
mskLib[254] = np.rot90(mskLib[239],2)
mskLib[255] = np.rot90(mskLib[239],3)

#Find best mask for each block
for i in range(len(T)):
    fnorm = 10000
    for j in range(len(mskLib)):
        if np.linalg.norm(T[i]-mskLib[j],'fro') <= fnorm:
            fnorm = np.linalg.norm(T[i]-mskLib[j],'fro')
            temp_coef = j
    print(temp_coef)
    coefC.append(temp_coef)
#End of compression. coefA,coefB, and coefC contain compressed data 
    
#Decompression
rblck = []
for i in range(len(coefA)):
    rblck.append((coefA[i]+(coefB[i]*mskLib[coefC[i]]/24)).round())
decomp = np.zeros([ary.shape[0],ary.shape[1]])
for i in range(np.shape(ary)[0]//4):
    for k in range(np.shape(ary)[1]//4):
            decomp[i*4:i*4+4,k*4:k*4+4] = rblck[i+np.shape(ary)[0]//4*k]

#Display images and compare to JPEG compression
newim = Image.fromarray(decomp.transpose())
newim.show()
print(compare_ssim(ary,decomp))
jpeg = Image.open('huugecatjpg.jpeg').convert('L')
print(compare_ssim(ary,np.array(jpeg)[0:(img.size[1]-img.size[1]%4),0:(img.size[0]-img.size[0]%4)].transpose()))
#jpeg.show()
#newim.convert('RGB').save(os.path.join('./imageafterchimeracomp.jpeg'),'PNG')
print(time.time()-startTime)








