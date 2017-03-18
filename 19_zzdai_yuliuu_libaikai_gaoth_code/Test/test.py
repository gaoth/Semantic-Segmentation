from __future__ import division
import numpy as np
from PIL import Image
from matplotlib.pyplot import *
import matplotlib
import os
import sys
from datetime import datetime
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import caffe


def hist_intersect(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

# load Net Parameters
print  '>>> Loading Net...'
net = caffe.Net('/scratch/eecs542w17_fluxg/zzdai/mytest/testfcn16s.prototxt', '/scratch/eecs542w17_fluxg/zzdai/mytest/train_iter_8000.caffemodel', caffe.TEST)
numClass = net.blobs['score'].channels


# Do Segmentation for Image List
# Read list for images required segmentation
seg = np.loadtxt('/scratch/eecs542w17_fluxg/zzdai/mytest/segList.txt', dtype=str)


for idx in seg:
    im = Image.open('/scratch/eecs542w17_fluxg/zzdai/data/pascal/VOC2011/JPEGImages/' + idx + '.jpg')
    in_ = np.array(im, dtype=np.float32)
    in_ = in_.transpose((2,0,1))
    #in_ = in_[:,:,::-1]
    #in_ -= np.array((104.00698793,116.66876762,122.67891434))

    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    out = net.blobs['score'].data[0].argmax(0).astype(np.uint8)
    im = Image.fromarray(out, mode='P')

    #Save Images to Local directory
    print 'Start Saving'
    im.save(os.path.join('/scratch/eecs542w17_fluxg/zzdai/Segmentations', idx + '_seg_16_2' + '.png'))
    print 'Finish Saving'


'''
np.set_printoptions(threshold='nan')
val = np.loadtxt('/scratch/eecs542w17_fluxg/zzdai/sample/mytest/valList.txt', dtype=str)
#print val
for idx in val:
    print idx
    im = Image.open('/scratch/eecs542w17_fluxg/zzdai/sample/data/pascal/VOC2011/JPEGImages/' + idx + '.jpg')
    gtru = Image.open('/scratch/eecs542w17_fluxg/zzdai/sample/data/pascal/VOC2011/SegmentationClass/' + idx + '.png')
    in_ = np.array(im, dtype=np.float32)
    in_ = in_.transpose((2,0,1))
    #in_ = in_[:,:,::-1]
    #in_ -= np.array((104.00698793,116.66876762,122.67891434))
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    out = net.blobs['score'].data[0].argmax(0).astype(np.uint8)
    im = Image.fromarray(out, mode='P')
    b = np.array(im,dtype = int)
    a = np.array(gtru, dtype = int)
    c = net.blobs['score'].data[0].argmax(0).flatten()
    k = (a >= 0) & (a < numClass)
    
    print np.bincount(a[k],minlength = 21)
    print np.bincount(b[k],minlength = 21)
    print np.bincount(c[k],minlength = 21)
#print np.bincount(b,minlength = 21)
#print b[k]
'''

