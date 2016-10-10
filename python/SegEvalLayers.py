import numpy as np 
import caffe

class SegIoULayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)

        # extract param class_num
        if 'class_num' not in params:
            raise Exception('"class_num" parameter must be specified')
        self.class_num = params['class_num']
        
        # two bottoms
        if len(bottom) < 2:
        	raise Exception('2 bottoms required, 1st for predictions, 2nd for labels')
        if len(bottom) > 2:
        	raise Exception('Only 2 bottoms required')
        
        # one top
        if len(top) < 1:
        	raise Exception('1 top required for the IoU result')
        if len(top) > 1:
        	raise Exception('Only 1 top required')

        # the result stored in a vector of size C + 1, one for averaging
        top[0].reshape(self.class_num + 2)

        # store two other tops for union and intersections
        self.union = np.zeros(self.class_num)
        self.inte = np.zeros(self.class_num)

    def reshape(self, bottom, top):
        # check two bottom blob have the same shape
        if bottom[0].data.shape != bottom[1].data.shape:
            raise Exception('two bottoms must have the same shape')

        # check bottom blob have only one channel
        if bottom[0].data.shape[1] != 1:
            raise Exception('Single channel is required. ' + \
                            'If the input is a probability map or one-hot codding, ' + \
                            'please use ArgMaxLayer with axis specified to produce a label map')
        
        # the number of image
        self.N = bottom[0].data.shape[0]
        
    def forward(self, bottom, top):
        # intersection and union of each class of each picture
    	for n in xrange(self.N):
            pred = bottom[0].data[n]
            label = bottom[1].data[n]
            for c in xrange(self.class_num):
                i = np.logical_and(pred == c, label == c).flatten().tolist().count(True)
                u = np.logical_or(pred == c, label == c).flatten().tolist().count(True)
                self.inte[c] += i
                self.union[c] += u
        
        # update IoU; if class c not present yet, set IoU = -1
        for c in xrange(self.class_num):
            if self.union[c] == 0:
                top[0].data[c] = 0
            else:
                top[0].data[c] = 1.0 * self.inte[c] / self.union[c]
        
        # compute average IoU, background class is ignored
        top[0].data[self.class_num] = top[0].data[1:].mean()
        top[0].data[self.class_num + 1] = 1.0 * self.inte[1:].sum() / self.union[1:].sum()

        # print the result to the screen
        for c in xrange(self.class_num):
            print 'IoU %d: %5f' % (c, top[0].data[c])
        print 'class mean IoU: %5f' % top[0].data[self.class_num]
        print 'pixel mean IoU: %5f' % top[0].data[self.class_num + 1]

    def backward(self, bottom, top):
    	pass
"""
class SegSavePngLayer(caffe.Layer) {
    def setup(self, bottom, top):
        params = eval(self.param_str)
        
        # directory to save the picture
        if 'save_dir' not in params:
            raise Exception('save_dir parameter must exist in parameters')
        save_dir = params['save_dir']

        # retrieve palette from a ground-truth picture
        if 'truth_ex' not in params:
            raise Exception('one ground-truth png image with color type 3 must exist to specify palette')
        truth_ex = params['truth_ex']

        # postfix
        postfix = params.get('postfix', '_result')

        if len(bottom) != 2:
            raise Exception('len of bottom:2, one bottom for result, one bottom for file name')
        if len(top) > 0:
            raise Exception('do not declare top blob')

    def reshape(self, bottom, top):
        

    def forward(self, bottom, top):
        pass
    def backward(self, bottom, top):
        pass
}"""
