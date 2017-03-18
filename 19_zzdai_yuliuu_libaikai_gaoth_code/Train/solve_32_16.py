import caffe
import transplant, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '/scratch/eecs542w17_fluxg/zzdai/train32_iter_100000.caffemodel'

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/scratch/eecs542w17_fluxg/zzdai/valList.txt', dtype=str)
f = open('/scratch/eecs542w17_fluxg/zzdai/Result32_16/fcn32_16result'+ str(ix) + '.txt','w')

for ix in range(100):
    solver.step(10000)
    print 'start'
    hist, loss = score.seg_tests(solver, False, val, layer='score',gt = 'label')
    f.write('\n'+str(ix)+'----------------------------------')
    #f.write('\n'+str(timeit.default_timer()))
    acc = np.diag(hist).sum() / hist.sum()
    f.write('\n overall accuracy:  '+str(acc))
    acc = np.diag(hist) / hist.sum(1)
    f.write('\n mean accuracy:  '+str(np.nanmean(acc)))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    f.write('\n mean IU:  '+str(np.nanmean(iu)))
    freq = hist.sum(1) / hist.sum()
    f.write('\n fwavacc:  ' +str((freq[freq > 0] * iu[freq > 0]).sum()))
    f.write('\n loss:  ' +str(loss))
