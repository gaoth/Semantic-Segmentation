import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def fcn(split):
	n = caffe.NetSpec()
	pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
			seed=1337)
	if split == 'train':
		pydata_params['sbdd_dir'] = '../data/pascal/VOC2011'
		pylayer = 'SBDDSegDataLayer'
	else:
		pydata_params['voc_dir'] = '../data/pascal/VOC2011'
		pylayer = 'VOCSegDataLayer'
	n.data, n.label = L.Python(module='voc_layers', layer=pylayer,
			ntop=2, param_str=str(pydata_params))
			
	# the base net
	n.conv1_1 = L.Convolution(n.data, kernel_size=3, stride=1, num_output=64, pad=100,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu1_1 = L.ReLU(n.conv1_1, in_place=True)
	n.conv1_2 = L.Convolution(n.relu1_1, kernel_size=3, stride=1, num_output=64, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu1_2 = L.ReLU(n.conv1_2, in_place=True)
	n.pool1 = L.Pooling(n.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
	
	n.conv2_1 = L.Convolution(n.pool1, kernel_size=3, stride=1, num_output=128, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu2_1 = L.ReLU(n.conv2_1, in_place=True)
	n.conv2_2 = L.Convolution(n.relu2_1, kernel_size=3, stride=1, num_output=128, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu2_2 = L.ReLU(n.conv2_2, in_place=True)
	n.pool2 = L.Pooling(n.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
	
	n.conv3_1 = L.Convolution(n.pool2, kernel_size=3, stride=1, num_output=256, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu3_1 = L.ReLU(n.conv3_1, in_place=True)
	n.conv3_2 = L.Convolution(n.relu3_1, kernel_size=3, stride=1, num_output=256, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu3_2 = L.ReLU(n.conv3_2, in_place=True)
	n.conv3_3 = L.Convolution(n.relu3_2, kernel_size=3, stride=1, num_output=256, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu3_3 = L.ReLU(n.conv3_3, in_place=True)
	n.pool3 = L.Pooling(n.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
	
	n.conv4_1 = L.Convolution(n.pool3, kernel_size=3, stride=1, num_output=512, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu4_1 = L.ReLU(n.conv4_1, in_place=True)
	n.conv4_2 = L.Convolution(n.relu4_1, kernel_size=3, stride=1, num_output=512, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu4_2 = L.ReLU(n.conv4_2, in_place=True)
	n.conv4_3 = L.Convolution(n.relu4_2, kernel_size=3, stride=1, num_output=512, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu4_3 = L.ReLU(n.conv4_3, in_place=True)
	n.pool4 = L.Pooling(n.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
	
	n.conv5_1 = L.Convolution(n.pool4, kernel_size=3, stride=1, num_output=512, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu5_1 = L.ReLU(n.conv5_1, in_place=True)
	n.conv5_2 = L.Convolution(n.relu5_1, kernel_size=3, stride=1, num_output=512, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu5_2 = L.ReLU(n.conv5_2, in_place=True)
	n.conv5_3 = L.Convolution(n.relu5_2, kernel_size=3, stride=1, num_output=512, pad=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu5_3 = L.ReLU(n.conv5_3, in_place=True)
	n.pool5 = L.Pooling(n.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)
	
	# fully connected
	n.fc6 = L.Convolution(n.pool5, kernel_size=7, stride=1, num_output=4096, pad=0,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu6 = L.ReLU(n.fc6, in_place=True)
	n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
	
	n.fc7 = L.Convolution(n.drop6, kernel_size=1, stride=1, num_output=4096, pad=0,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.relu7 = L.ReLU(n.fc7, in_place=True)
	n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
	
	n.score_fr = L.Convolution(n.drop7, num_output=21, kernel_size=1, pad=0, stride=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.upscore2 = L.Deconvolution(n.score_fr,
		convolution_param=dict(num_output=21, kernel_size=4, stride=2, bias_term=False),
		param=[dict(lr_mult=0)])
	
	n.score_pool4 = L.Convolution(n.pool4, num_output=21, kernel_size=1, pad=0,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.score_pool4c = crop(n.score_pool4, n.upscore2)
	n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c, operation=P.Eltwise.SUM)
	n.upscore_pool4 = L.Deconvolution(n.fuse_pool4,
		convolution_param=dict(num_output=21, kernel_size=4, stride=2, bias_term=False),
		param=[dict(lr_mult=0)])
	
	n.score_pool3 = L.Convolution(n.pool3, num_output=21, kernel_size=1, pad=0,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	n.score_pool3c = crop(n.score_pool3, n.upscore_pool4)
	n.fuse_pool3 = L.Eltwise(n.upscore_pool4, n.score_pool3c, operation=P.Eltwise.SUM)
	n.upscore8 = L.Deconvolution(n.fuse_pool3,
		convolution_param=dict(num_output=21, kernel_size=16, stride=8, bias_term=False),
		param=[dict(lr_mult=0)])
	
	n.score = crop(n.upscore8, n.data)
	n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_param=dict(normalize=True, ignore_label=255))
	
	return n.to_proto()
	
def make_net():
	with open('train.prototxt', 'w') as f:
		f.write(str(fcn('train')))

	with open('val.prototxt', 'w') as f:
		f.write(str(fcn('seg11valid')))
		
if __name__ == '__main__':
	make_net()
	
