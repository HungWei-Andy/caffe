import numpy as np

def IoU(blob1, blob2, clsN):
  print 'blob1 shape: ', blob1.shape
  print 'blob2 shape: ', blob2.shape
  if blob1.shape != blob2.shape:
    print 'Two Blobs does not have same shape'
    return

  # parameter
  res = np.zeros(clsN)  

  # compare IoU of each channel
  for c in xrange(clsN):
    inte = np.logical_and(blob1 == c, blob2 == c).flat.tolist().count(True)
    uni = np.logical_or(blob1 == c, blob2 == c).flat.tolist().count(True)
    res[c] = 1.0 * inte / uni
  return res
