# Merge Batch Normalization in caffe
This implementation is about a fusion of batch normalization with convolution or fully connected layers in CNN of [Caffe](https://github.com/BVLC/caffe).


## Introduction
Caffe uses two layers to implement bn:

```
layer {
  name: "conv1-bn"
  type: "BatchNorm"
  bottom: "conv1"
  top: "conv1"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  batch_norm_param {
    moving_average_fraction: 0.99
    eps: 1e-8
  }
}
layer {
  name: "conv1-bn-scale"
  type: "Scale"
  bottom: "conv1"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 0
  }
  param {
    lr_mult: 1
    decay_mult: 1
  }
  scale_param {
    axis: 1
    num_axes: 1
    filler {
      type: "constant"
      value: 1
    }
    bias_term: true
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}    
```

When a model training is finished, both batch norm and scale layer learn their own parameters, these parameters are fixed during inference. So, we can merget it with the convolution or fully connected layer.

For MORE details about batch normalization，see [here](https://arxiv.org/abs/1502.03167)

## Demo

#### Note: 
The network of the demo contains the ROI layer, make sure u are in a Pre-compiled environment of [fasterRCNN](https://github.com/rbgirshick/py-faster-rcnn). If U want to convert your own model, you can put the script into your own environment.


RUN
``
python convert_2_nonbnn.py   
``
to convert the normal network to the one without bn.

RUN
``
python test_convert.py
``
to test the demo network.



