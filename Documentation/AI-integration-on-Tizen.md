---
title: AI-Integration-on-Tizen
...

# Use AI on Tizen

## Briefing

This guide is for Tizen developers who want to use AI in their application. First, we introduce [`ai-edge-torch`](https://github.com/google-ai-edge/ai-edge-torch), a python package tool converting [PyTorch](https://pytorch.org/) model to [TensorFlow Lite (TFLite)](https://ai.google.dev/edge/lite) model. Then, we will introduce how to use TFLite model in Tizen using [Machine Learning Service APIs](https://docs.tizen.org/application/native/guides/machine-learning/machine-learning-service/).

PyTorch is great for training and developing AI models. However, it's not so easy to deploy and use it in on-device. TFLite has great on-device Tizen performance and it's easy to use GPU, DSP, or NPU by its delegate feature. Thus we recommend using TFLite in Tizen, at least for PoC stage (after engineering you may find better solution).

## Prepare your model

First, you should obtain your model. There are several ways:

1. Check [NNStreamer/ML-API Tizen Examples](https://github.com/nnstreamer/nnstreamer-example/)
    - There are many real examples you may find it useful.
2. Find web. There are many open models in the web.
    - [Official TFLite examples](https://ai.google.dev/edge/lite#1_generate_a_tensorflow_lite_model)
    - [ultralytics](https://docs.ultralytics.com/models/) offers many YOLO based vision models.
    - [huggingface](https://huggingface.co/) or [torchvision models](https://pytorch.org/vision/stable/models.html)
3. Make your own model (PyTorch)

    In many cases, you need to use your own model. This model might be written from scratch or trained with your custom dataset. If your model is made with PyTorch, you can use [ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch) to convert your model into TFLite format, so it can be used in Tizen.

### Make PyTorch model

For brevity, this article use a pretrained [MobileNet V3 small](https://pytorch.org/vision/main/models/mobilenetv3.html) model from torchvision.

```python
import torch
import torchvision

# Here we use mobilenet_v3_small with pre-trained weights.
torch_model = torchvision.models.mobilenet_v3_small(torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()

# Finetune with your own data if you want.

sample_input = torch.randn(1, 3, 224, 224) # All pytorch CONV layer has NCHW format
sample_input.shape # torch.Size([1, 3, 224, 224])

sample_output = torch_model(sample_input)
saample_output.shape # torch.Size([1, 1000])
```

Obviously, you could/should finetune the model with your own data!

### Convert PyTorch into TFLite with [ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch)

[ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch) is Google's open source project to convert PyTorch model into TFLite model. Check [official guide](https://github.com/google-ai-edge/ai-edge-torch/blob/main/docs/pytorch_converter/README.md) for more details.

1. [Install ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch?tab=readme-ov-file#installation)

    ```bash
    pip install -r https://raw.githubusercontent.com/google-ai-edge/ai-edge-torch/main/requirements.txt
    pip install ai-edge-torch-nightly
    ```

2. Export PyTorch model to TFLite model file.

    ```python
    import ai_edge_torch

    edge_model = ai_edge_torch.convert(torch_model, (sample_input,)) # convert the model
    edge_model_output = edge_model(sample_input) # run the model
    edge_model_output.shape # (1, 1000)
    edge_model.export("mobilenet_v3_small.tflite") # save as tflite model file
    ```

3. Validate the TFLite model.

    ```python
    import numpy as np
    import tensorflow as tf

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="mobilenet_v3_small.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    #  'name': 'serving_default_args_0:0',
    #  'index': 0,
    #  'shape': array([  1,   3, 224, 224], dtype=int32),
    #  'shape_signature': array([  1,   3, 224, 224], dtype=int32),
    #  'dtype': numpy.float32,
    #  'quantization': (0.0, 0),
    #  'quantization_parameters': {'scales': array([], dtype=float32),
    #    'zero_points': array([], dtype=int32),
    #    'quantized_dimension': 0},
    #   'sparsity_parameters': {}}


    output_details = interpreter.get_output_details()
    #  'name': 'StatefulPartitionedCall:0',
    #  'index': 254,
    #  'shape': array([   1, 1000], dtype=int32),
    #  'shape_signature': array([   1, 1000], dtype=int32),
    #  'dtype': numpy.float32,
    #  'quantization': (0.0, 0),
    #  'quantization_parameters': {'scales': array([], dtype=float32),
    #    'zero_points': array([], dtype=int32),
    #    'quantized_dimension': 0},
    #  'sparsity_parameters': {}

    # Test the model on random input data.
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    ```

Now you have tflite model file `mobilenet_v3_small.tflite`. You can use it in your Tizen app with ML Service APIs.

## Use [ML Service API](https://docs.tizen.org/application/native/guides/machine-learning/machine-learning-service/) on Tizen

Tizen ML API provides a set of APIs for AI applications to use its model in a convenient conf-file-based passion.

### The conf file

A conf file `mobilenet.conf` should be something like this:

```json
{
  "single" : {
    "model" : "mobilenet_v3_small.tflite",
    "framework" : "tensorflow-lite",
    "custom" : "Delegate:XNNPACK,NumThreads:2",
    "input_info" : [
      {
        "type" : "float32",
        "dimension" : "224:224:3:1"
      }
    ],
    "output_info" : [
      {
        "type" : "float32",
        "dimension" : "1000:1"
      }
    ]
  },
  "information" :
  {
    "label_file" : "mobilenet_label.txt"
  }
}
```

- `"model" : "mobilenet_v3_small.tflite"` means this conf file uses the specified model.
- `"framework" : "tensorflow-lite"` means this model should be invoked by TFLite.
- `"custom" : "Delegate:XNNPACK,NumThreads:2"` sets custom options to the model.
- `"input_info"` and `"output_info"` are used to specify input and output tensor information. This can be omitted.
- `"information"` defines other information using this ml-service. Here label file path is specified.

### Usage of ML Service API

A ml_service handle is created by API `ml_service_new`

```C
ml_service_h mls;
char *conf_file_path = "path/to/mobilenet.conf";
ml_service_new (conf_file_path, &mls);
```

Check input and output of `mls`

```C
ml_tensors_info_h input_info;
ml_service_get_input_information (mls, NULL, &input_info);
/*
- tensor count: 1
- tensor[0]
  - name: serving_default_args_0:0
  - type: 7
  - dimension[0]: 224
  - dimension[1]: 224
  - dimension[2]: 3
  - dimension[3]: 1
  - size: 150528 byte
*/

ml_tensors_info_h output_info;
ml_service_get_output_information (mls, NULL, &output_info);
/*
- tensor count: 1
- tensor[0]
  - name: StatefulPartitionedCall:0
  - type: 7
  - dimension[0]: 1000
  - dimension[1]: 1
  - dimension[2]: 0
  - dimension[3]: 0
  - size: 4000 byte
*/
```

Set callback. ML Service API provides async method for getting result value of `mls`.

```C
void
_new_data_cb (ml_service_event_e event, ml_information_h event_data, void *user_data)
{
  ml_tensors_data_h new_data = NULL;

  if (event != ML_SERVICE_EVENT_NEW_DATA)
    return;

  // get tensors-data from event data.
  ml_information_get (event_data, "data", &new_data);

  // get the float result
  float *result;
  size_t result_size;
  ml_tensors_data_get_tensor_data (new_data, 0U, (void *) &result, &result_size);

  // result : float[1000]
  // result_size : 4000 byte
  // do something useful with the result.
}
...

// set event callback
ml_service_set_event_cb (mls, _new_data_cb, user_data);
```

All setup is done. Let's invoke `mls` with proper input data with API `ml_service_request`

```C
ml_tensors_data_h input_data = NULL;
ml_tensors_data_create (input_info, &input_data); /* create input data layout of input_info */

// set data_buf as you want.
// It should be float value image data float[224][224][3][1].
uint8_t *data_buf = NULL;
size_t data_buf_size;

// set 0-th tensor data with user given buffer
ml_tensors_data_set_tensor_data (input_data, 0U, data_buf, data_buf_size);

// now request mls to invoke the model with given input_data
ml_service_request (mls, NULL, input_data);

// When mls get the result of model inference, `_new_data_cb` will be called.
```

### Package your `mls` into [RPK](https://docs.tizen.org/application/tizen-studio/native-tools/rpk-package)

[Tizen Resource Package (RPK)](https://docs.tizen.org/application/tizen-studio/native-tools/rpk-package) is a package type dedicated to resources. Tizen ML Service API utilizes RPK to let app developers easily decouple their ML from their application, and upgrade their model without re-packaging/deploying their application. Please check [this real example](https://github.com/nnstreamer/nnstreamer-example/tree/main/Tizen.native/ml-service-example) for details.
