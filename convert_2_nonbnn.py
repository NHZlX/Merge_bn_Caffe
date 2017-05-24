#coding=utf-8

import os.path as osp
import sys
import copy
import os
import numpy as np
import google.protobuf as pb

CAFFE_ROOT = './'
if osp.join(CAFFE_ROOT,'python') not in sys.path:
        sys.path.insert(0,osp.join(CAFFE_ROOT,'python'))

import caffe
import caffe.proto.caffe_pb2 as cp

caffe.set_mode_cpu()
layer_type = ['Convolution', 'InnerProduct']
bnn_type = ['BatchNorm', 'Scale']
temp_file = './temp.prototxt'

class ConvertBnn:
    def __init__(self, model, weights, dest_model_dir, dest_weight_dir):
        self.net_model = caffe.Net(model, weights, caffe.TEST)
        self.net_param = self.get_netparameter(model)
        self.dest_model = None
        self.dest_param = self.get_netparameter(model)
        self.remove_ele = []
        self.bnn_layer_location = []
        self.dest_dir = dest_model_dir
        self.dest_weight_dir = dest_weight_dir
        self.pre_process()
        
    def pre_process(self):
        net_param = self.dest_param
        layer_params = net_param.layer
        length = len(layer_params)
        i = 0
        while i < length:
            print i
            if layer_params[i].type in layer_type:
                if (i + 2 < length) and layer_params[i + 1].type == bnn_type[0] and  \
                    layer_params[i + 2].type == bnn_type[1]:
                        params = layer_params[i].param
                        if len(params) < 2:
                            params.add()
                            params[1].lr_mult = 2 
                            params[1].decay_mult = 0
                            layer_params[i].convolution_param.bias_term = True
                            layer_params[i].convolution_param.bias_filler.type = 'constant'
                            layer_params[i].convolution_param.bias_filler.value = 0
                        #修改配置params
                        self.bnn_layer_location.extend([i, i + 1, i + 2])
                        self.remove_ele.extend([layer_params[i + 1], layer_params[i + 2]])
                        i = i + 3
                else:
                    i += 1
            else:
                i += 1
        
        #for ele in remove_ele:
        #    layer_params.remove(ele)
        with open(temp_file, 'w') as f:
            f.write(str(net_param))
        print 'asdf'
        self.dest_model = caffe.Net(temp_file, caffe.TEST)
        model_layers = self.net_model.layers
        for i, layer in enumerate(model_layers):
            if layer.type == 'Convolution' or layer.type == 'InnerProduct':
                self.dest_model.layers[i].blobs[0] = layer.blobs[0]
                if len(layer.blobs) > 1:
                    self.dest_model.layers[i].blobs[1] = layer.blobs[1]
        print 'asdf end'
    
    def get_netparameter(self, model):
        with open(model) as f:
            net = cp.NetParameter()
            pb.text_format.Parse(f.read(), net)
            return net

    def convert(self):
        #layer param 需要修改 BIAS 参数 添加bias param 还有设置为 true
        out_params = self.dest_param.layer
        model_layers = self.net_model.layers
        out_model_layers = self.dest_model.layers
        
        print self.bnn_layer_location
        length = len(self.bnn_layer_location)
        l = 0
        while l < length:
            i = self.bnn_layer_location[l]
            channels = model_layers[i].blobs[0].num
            #count = model_layers[i].blobs[0].count / channels
            scale = model_layers[i + 1].blobs[2].data[0]
            print scale
            mean = model_layers[i + 1].blobs[0].data / scale
            print mean
            std = np.sqrt(model_layers[i + 1].blobs[1].data / scale)
            a = model_layers[i + 2].blobs[0].data
            b = model_layers[i + 2].blobs[1].data
            for k in xrange(channels):
                out_model_layers[i].blobs[0].data[k] = model_layers[i].blobs[0].data[k] * a[k] / std[k]
                out_model_layers[i].blobs[1].data[k] = out_model_layers[i].blobs[1].data[k] * a[k] / std[k] - a[k] * mean[k] / std[k] + b[k] 
            l += 3
        self.dest_model.save(self.dest_weight_dir)
        for ele in self.remove_ele:
            out_params.remove(ele)
        with open(self.dest_dir, 'w') as f:
            f.write(str(self.dest_param))
        os.remove(temp_file)

if __name__ == '__main__':
    cb = ConvertBnn('./models/deploy_68_new.prototxt', './models/vgg_68_new.caffemodel', './models/result.prototxt', './models/result.caffemodel')
    cb.convert()
