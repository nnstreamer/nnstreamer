---
title: Edge-AI / Among-Device AI
...

# What is Edge-AI and Among-Device AI?


With on-device AI, different embedded devices including mobile phones, TVs, refrigerators, vacuum cleaners, and all the other sorts of deviecs have started running deep neural network models.
Unlike conventional AI services running in cloud servers, on-device AI services are often limited by the limited computing resources of the given devices; they usually have limited size of memory, limited processing power, or limited energy, which varies per device.
Moreover, such devices have different availability of data; each device may have different sensors or different sensing targets (e.g., view angles of cameras), which might be beneficial if a device has access to others.


Among-Device AI (a.k.a. Edge-AI in some industry context) tries to mitigate such issues by connecting such devices and let them share data and processing power between them.
In other words, Among-Device AI / Edge-AI allows distributed computing for AI algorithms, which can distribute computing workloads and share data from different devices.
Running AI at the ‘edge’ of the local network removes the requirement for the device to be connected to the internet or centralized servers like the cloud.
Edge AI offers significant improvements as far as response speeds and data security.
Executing AI close to the data source allows for processes like data creation and decision-making to take place in milliseconds, making Edge AI ideal for applications where near-instantaneous responses are essential.


Among-Device AI / Edge-AI can be often extended to conventional cloud AI services so that parts of data can be sent to clouds for further processing or data gathering for future neural network models.
Technically, for pipeline frameworks such as NNStreamer and GStreamer, they are not very different from among-device AI between embedded devices (exactly same code can be applied!).


In the NNStreamer's point of view, among-device AI can be achieved by connecting NNStreamer pipelines running in different devices.
By connecting pipelines of different devices, devices can send data for inferences to other devices and receive inference results or data from sensors of other devices.
Moreover, by completing #3745, this can be expanded to federated learning and training offloading, which is expected to be enabled by 2023.


  * Reference
    - [On the Edge - How Edge AI is reshaping the future](https://www.samsung.com/semiconductor/newsroom/tech-trends/on-the-edge-how-edge-ai-is-reshaping-the-future/), Samsung Newsroom
    - [Toward Among-Device AI from On-Device AI with Stream Pipelines](https://arxiv.org/abs/2201.06026), ICSE 2022 SEIP


# Edge-AI Example Applications with NNStreamer
Note : If you are new to NNStreamer, see [usage examples screenshots](https://github.com/nnstreamer/nnstreamer/wiki/usage-examples-screenshots).

Examples show how to implement edge AI with NNStreamer.

These examples are tested on Ubuntu PC and Raspberry PI.

  * [Image Classification](https://github.com/nnstreamer/nnstreamer-example/tree/main/Tizen.platform/Tizen_IoT_ImageClassification)

    The device analyzes the camera image before transmitting it, and then transmits meaningful information only.

    In this example, if the device finds a target that the user wants, it starts video streaming to the server.
  * [Text Classification](https://github.com/nnstreamer/nnstreamer-example/tree/main/Tizen.platform/Tizen_IoT_text_classification_NonGUI)

    Text classifications are classified into predefined groups based on sentences.

  * [Image segmentation on edgeTPU](https://github.com/nnstreamer/nnstreamer-example/tree/main/bash_script/example_image_segmentation_tensorflow_lite)

    Image segmentation is the process of partitioning a digital image into multiple segments.

    This application shows how to send the flatbuf to the edge device and run inferences on the edgeTPU.


### High level deployment diagram of NNStreamer edge-AI examples

* [deployment diagram](./media/edgeai_diagram.png): The picture is misleading as each instance of "edge-AI" does NOT require clouds although it may utilize clouds. It should be redrawn.

*To help understand the three edge-AI examples, it would be different from the actual.
