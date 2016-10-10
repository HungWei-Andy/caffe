import numpy as np
from PIL import Image

import evalres
import caffe

img_dir = '/home/andy/data/pascal/VOC2012/JPEGImages/2007_000039.jpg'
truth_dir = '/home/andy/data/pascal/VOC2012/SegmentationClass/2007_000039.png'
file_dir = '/home/andy/caffe/models/fcn/voc-vgg32s/'
val_file = file_dir + 'train.prototxt'
model_file = file_dir + 'snapshot/train_iter_19500.caffemodel'

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open(img_dir)
im.show()
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load an arbitrary result to get palette
truthImg= Image.open(truth_dir)
truthImg.show()
pal = truthImg.getpalette()

# load net
net = caffe.Net(val_file, model_file, caffe.TEST)

# shape for input (data blob is N x C x H x W), set data
#net.blobs['data'].reshape(1, *in_.shape)
#net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
#print im.size
#print in_.shape
#print truthImg.size
#print net.blobs['data'].data[0].shape
#print net.blobs['data'].data[...]
#print net.blobs['score'].data[0].shape
out = net.blobs['pred'].data[0, 0]
print out[185: 195, 245: 255]
print out.shape
#print out.flatten().tolist().count(0), out.shape[0] * out.shape[1]
#print out.shape
#print out

label = net.blobs['label'].data[0, 0]
print label[185: 195, 245: 255]
print label.shape

# transform ground truth from one-hot to index
#print evalres.IoU(truth, out, 21)

# format the result as a graph
res_img = Image.fromarray(out.astype(np.uint8), mode='P')
res_img.putpalette(pal)
res_img.show()

lab_img = Image.fromarray(out.astype(np.uint8), mode='P')
lab_img.putpalette(pal)
lab_img.show()

