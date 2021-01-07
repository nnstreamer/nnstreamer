---
title: Writing Tizen C# apps
...

# Writing a Tizen .NET Application

This document provides you how to write a Tizen .NET Application with Machine Learning APIs.
Since Tizen 5.5, Machine Learning Inference functionality has been provided on Mobile, Wearable and TV profile.

## Installing Visual Studio Tools for Tizen
In order to use this functionality, you need to install Visual Studio Tools for Tizen and Tizen SDK. You can find the detailed guide for this at the below link.

* https://docs.tizen.org/application/vstools/install


## Machine Learning

Machine learning (ML) inference feature introduces how you can easily invoke the neural network model and get the inference output result effortlessly and efficiently.

You can use the following machine learning feature in your .NET applications:

You can use the `Tizen.MachineLearning.Inference.SingleShot` class, to load the existing neural network model or your own specific model from the storage. After loading the model, you can invoke it with a single instance of input data. Then, you can get the inference output result.

You can also use the `Pipeline` feature to manage the topology of data and the interconnection between processors and models. This feature is available in Native APIs from Tizen 5.5. However, this feature is not available in .NET APIs. This feature will be available in the .NET APIs from the next Tizen version.


The main features of the Tizen.MachineLearning.Inference are:

- Managing tensor information, which is the metadata: dimensions and types of tensors

  You can [configure the input and output Tensor Information](#manage) such as its name, data type and dimension.

- Loading a neural network model and configuring a runtime environment

  You can [load the neural network model from storage and configure a runtime environment](#load).

- Invoking the neural network model with input data

  After setting up the SingleShot instance with its required information, you can [invoke the model with the input data and get the inference output result](#invoke).

- Fetching the inference result after invoking

  You can [fetch the inference result](#fetch) after invoking the respective model.

## Prerequisites

To enable your application to use the Machine Learning Inference API functionality:

1. To use the methods and properties of the `Tizen.MachineLearning.Inference.SingleShot` class or its related classes such as `Tizen.MachineLearning.Inference.TensorsData` and `Tizen.MachineLearning.Inference.TensorsInfo`, include the `Tizen.MachineLearning.Inference` namespace in your application:
    ```C#
    using Tizen.MachineLearning.Inference;
    ```

2. If the model file you want to use is located in the **media storage** or the **external storage**, the application has to request permission by adding the following privileges to the `tizen-manifest.xml` file:

    ```xml
    <privileges>
      <!--To access media storage-->
      <privilege>http://tizen.org/privilege/mediastorage</privilege>

      <!--To access, read, and write to the external storage-->
      <privilege>http://tizen.org/privilege/externalstorage</privilege>
    </privileges>
    ```

<a name="manage"></a>
## Managing Tensor Information

In the example mentioned in this page, the MobileNet v1 model for TensorFlow Lite is used. This model is used for image classification. The input data type of the model is specified as bit width of each Tensor and its input dimension is `3 X 224 X 224`. The output data type of the model is the same as the input datatype but the output dimension is `1001 X 1 X 1 X 1`.

To configure the tensor information, you need to create a new instance of the `Tizen.MachineLearning.Inference.TensorsInfo` class. Then, you can add the tensor information such as datatype, dimension, and name (optional) as shown in the following code:

```C#
/* Input Dimension: 3 * 224 * 224 */
TensorsInfo in_info = new TensorsInfo();
in_info.AddTensorInfo(TensorType.UInt8, new int[4] { 3, 224, 224, 1 });

/* Output Dimension: 1001 for classification */
TensorsData out_info = new TensorsInfo();
out_info.AddTensorInfo(TensorType.UInt8, new int[4] { 1001, 1, 1, 1 });
```

<a name="load"></a>
## Loading Neural Network Model and Configuring Runtime Environment

1. Since the model file is located in the resource directory of your own application, you need to get its absolute path:

    ```C#
    string ResourcePath = Tizen.Applications.Application.Current.DirectoryInfo.Resource;
    string model_path = ResourcePath + "models/mobilenet_v1_1.0_224_quant.tflite";
    ```


2. You can load the neural network model from storage and configure a runtime environment with the `Tizen.MachineLearning.Inference.SingleShot` class. The first parameter is the absolute path to the neural network model file. The remaining two parameters are the input and the output `TensorsInfo` instances. If there is an invalid parameter, `ArgumentException` is raised:

    ```C#
    /* Create SingleShot instance with model information */
    SingleShot single = new SingleShot(model_path, in_info, out_info);
    ```

<a name="invoke"></a>
## Invoking Neural Network Model using Input Data

To invoke the neural network model, you need to create the `Tizen.MachineLearning.Inference.TensorsData` instance to pass the input data of the model. You can add various types of tensor data, which are already specified in the `TensorInfo` instance. However, the maximum size of `TensorsData` is 16. If the limit is exceeded, then `IndexOutOfRangeException` is raised. Input data is passed in a byte array format, byte[]:

```C#
/* Input data for test */
byte[] in_buffer = new byte[3 * 224 * 224 * 1];

/* Set the input tensor data */
TensorsData in_data = in_info.GetTensorsData();
in_data.SetTensorData(0, in_buffer);
```

After preparing the input data, you can invoke the model and get the inference output result. The `SingleShot.Invoke()` method gets the input data to be inferred as a parameter and returns the `Tizen.MachineLearning.InferenceTensorsData` instance, which contains the inference result:

```C#
/* Invoke the model and get the inference result */
TensorsData out_data = single.Invoke(in_data);
```

<a name="fetch"></a>
## Fetching Inference Result

After calling the `Invoke()` method of the `Tizen.MachineLearning.Inference.SingleShot` class,
the `Tizen.MachineLearning.Inference.TensorsData` instance is returned as the inference result.
The result can have multiple output data. Therefore, you have to fetch each data using the `GetTensorData()` method. If the limit is exceeded, then `IndexOutOfRangeException` is raised:


```C#
/* Get the first Tensor data from the inference result */
byte[] out_buffer = out_data.GetTensorData(0);
```

The `TensorsData` class is used to send the input data to a neural network model. In addition, it provides the `Count` property to get the number of tensors:

```C#
/* Get the number of Tensor in TensorsData instance */
var count = out_data.Count;
```

## Related Information
- Dependencies
  -   Tizen 5.5 and Higher
