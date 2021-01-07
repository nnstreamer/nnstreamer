---
title: vivante
...

This describes how to enable the NNstreamer tensor filter as a sub-plugin for Vivante NPU.

## Vivante tensor filter
The entire procedure is as follows:
 * Read multi-files (e.g., vivante_v3.nb and libvivante_v3.so)
 * Create a neural network
 * Execute pre-process for the image data
 * Verify a graph
 * Process the graph
 * Dump all node outputs (optionally, for a debugging)
 * Do post-process for output data
 * Release the neural network

## Reference
 * http://www.vivantecorp.com/
 * https://www.khadas.com/product-page/vim3 (Amlogic A311D with 5.0 TOPS NPU)

## How to buid
First of all, you must generate a library of a model (e.g., libvivantev3.so, libyolov3.so) to use NNStreamer tensor filter.
For more details, please refer to the below repository.
 * https://www.github.com/nnstreamer/reference-models (TODO) - Press 'models' folder.

### Enable Vivante tensor filter
Since the Vivante tensor filter depends on the low-level libraries of Vivante NPU Toolkit,
the Vivante tensor filter is disabled by default. If you want to enable the Vivante tensor filter,
If you want to enable the Vivante tensor filter in Tizen, you need to modify the "vivante_support 0" statement in the ./packaging/nnstreamer.spec file.
For Ubuntu/ARM, you can compile the Vivante tensor filter source code with the ./build_run.sh file on the target.

### On VIM3/Ubuntu18.04 (arm64)
Please append the APT repository as follows for VIM3/Ubuntu 18.04 target board.
```bash
vim3$ sudo /usr/sbin/ntpdate jp.pool.ntp.org
vim3$ sudo apt-get update -o Acquire::https::dl.khadas.com::Verify-Peer=false
vim3$ cat /etc/apt/sources.list.d/fenix.list
deb [trusted=yes] https://dl.khadas.com/repos/vim3/ bionic main
```

Build and run the NNStreamer tensor filter on Ubuntu18.04 (arm64) + the VIM3/Khadas board.

```bash
vim3$ git clone https://github.com/nnstreamer/nnstreamer.git
vim3$ cd ./nnstreamer/ext/nnstreamer/tensor_filter/vivante/
vim3$ ./build_run.sh
```

### On Tizen/Unified (aarch64)
Build and run the NNStreamer tensor filter on Tizen (aarch64).

```bash
$ vi ~/.gbs.conf
$ gbs build -A aarch64 --clean --include-all
```

## How to run
You can get the below result when the required packages are prepared on the target board.


#### Case study: Run inceptionv3 model on the Ubuntu18.04 + VIM3 NPU board
The below log message show the execution result of the NNStreamer pipeline on both VIM3/Ubuntu18.04 (aarch64) board.
Please refer to the "**gst-launch-1.0 ....**" statement in the [./build_run.sh](./build_run.sh) file.

```bash
ubuntu$ export VSI_NN_LOG_LEVEL={0...5}
ubuntu$ gst-launch-1.0 filesrc location=/usr/share/dann/sample_pizza_299x299.jpg ! jpegdec ! videoconvert ! video/x-raw,format=RGB,width=299,height=299 ! tensor_converter ! tensor_filter framework=vivante model="/usr/share/dann/inception-v3.nb,/usr/share/vivante/inceptionv3/libinceptionv3.so" ! filesink location=vivante.out.bin

(gst-launch-1.0:4587): GLib-CRITICAL **: 00:43:47.896: g_file_test: assertion 'filename != NULL' failed
D [setup_node:367]Setup node id[0] uid[0] op[NBG]
D [print_tensor:129]in : id[   1] shape[ 3, 299, 299, 1   ] fmt[u8 ] qnt[ASM zp=137, scale=0.007292]
D [print_tensor:129]out: id[   0] shape[ 1001, 1          ] fmt[f16] qnt[NONE]
D [optimize_node:311]Backward optimize neural network
D [optimize_node:318]Forward optimize neural network
I [compute_node:260]Create vx node
Create Neural Network: 30ms or 30108us
I [vsi_nn_PrintGraph:1308]Graph:
I [vsi_nn_PrintGraph:1309]***************** Tensors ******************
D [print_tensor:137]id[   0] shape[ 1001, 1          ] fmt[f16] qnt[NONE]
D [print_tensor:137]id[   1] shape[ 3, 299, 299, 1   ] fmt[u8 ] qnt[ASM zp=137, scale=0.007292]
I [vsi_nn_PrintGraph:1318]***************** Nodes ******************
I [vsi_nn_PrintNode:156](             NBG)node[0] [in: 1 ], [out: 0 ] [a6981690]
I [vsi_nn_PrintGraph:1327]******************************************
Setting pipeline to PAUSED ...
Pipeline is PREROLLING ...
Start run graph [1] times...
Run the 1 time: 33ms or 33170us
vxProcessGraph execution time:
Total   33ms or 33244us
Average 33.24ms or 33244.00us
I [vsi_nn_ConvertTensorToData:732]Create 2002 data.
 --- Top5 ---
208: 0.824707
209: 0.044342
223: 0.008614
268: 0.002846
185: 0.002605
I [vsi_nn_ConvertTensorToData:732]Create 2002 data.
Pipeline is PREROLLED ...
Setting pipeline to PLAYING ...
New clock: GstSystemClock
Got EOS from element "pipeline0".
Execution ended after 0:00:00.000172250
Setting pipeline to PAUSED ...
Setting pipeline to READY ...
Setting pipeline to NULL ...
Freeing pipeline ...


khadas@Khadas:~/nnstreamer-private-plugins$ ls -al ./vivante.out.bin
-rw-rw-r-- 1 khadas khadas 2002 Jan 30 00:43 ./vivante.out.bin
```

#### Case study: Run Yolov3 model on the Ubuntu18.04 + VIM3 NPU board
Please refer to the "**gst-launch-1.0 ....**" statement in the [./build_run.sh](./build_run.sh) file.
```bash
ubuntu$ export VSI_NN_LOG_LEVEL={0...5}
ubuntu$ gst-launch-1.0 filesrc location=/usr/share/dann/sample_car_bicyle_dog_416x416.jpg ! jpegdec ! videoconvert ! video/x-raw,format=BGR,width=416,height=416 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=typecast option=int8 ! tensor_filter framework=vivante model="/usr/share/dann/yolov3.nb,/usr/share/vivante/yolov3/libyolov3.so" ! filesink location=vivante.out.bin

D [setup_node:368]Setup node id[0] uid[0] op[NBG]
D [print_tensor:136]in(0) : id[   0] vtl[0] const[0] shape[ 416, 416, 3, 1   ] fmt[i8 ] qnt[DFP fl=  7]
D [print_tensor:136]out(0): id[   1] vtl[0] const[0] shape[ 13, 13, 255, 1   ] fmt[i8 ] qnt[DFP fl=  2]
D [print_tensor:136]out(1): id[   2] vtl[0] const[0] shape[ 26, 26, 255, 1   ] fmt[i8 ] qnt[DFP fl=  2]
D [print_tensor:136]out(2): id[   3] vtl[0] const[0] shape[ 52, 52, 255, 1   ] fmt[i8 ] qnt[DFP fl=  2]
D [optimize_node:312]Backward optimize neural network
D [optimize_node:319]Forward optimize neural network
I [compute_node:261]Create vx node
Create Neural Network: -179140096ms or 41us
I [vsi_nn_PrintGraph:1421]Graph:
I [vsi_nn_PrintGraph:1422]***************** Tensors ******************
D [print_tensor:146]id[   0] vtl[0] const[0] shape[ 416, 416, 3, 1   ] fmt[i8 ] qnt[DFP fl=  7]
D [print_tensor:146]id[   1] vtl[0] const[0] shape[ 13, 13, 255, 1   ] fmt[i8 ] qnt[DFP fl=  2]
D [print_tensor:146]id[   2] vtl[0] const[0] shape[ 26, 26, 255, 1   ] fmt[i8 ] qnt[DFP fl=  2]
D [print_tensor:146]id[   3] vtl[0] const[0] shape[ 52, 52, 255, 1   ] fmt[i8 ] qnt[DFP fl=  2]
I [vsi_nn_PrintGraph:1431]***************** Nodes ******************
I [vsi_nn_PrintNode:159](             NBG)node[0] [in: 0 ], [out: 1, 2, 3 ] [ab9fecf0]
I [vsi_nn_PrintGraph:1440]******************************************
[DEBUG] input_tensors_num :1
[DEBUG] output_tensors_num:3
[DEBUG] input_dim_num[0]:4
[DEBUG] output_dim_num[0]:4
[DEBUG] output_dim_num[1]:4
[DEBUG] output_dim_num[2]:4
Setting pipeline to PAUSED ...
Pipeline is PREROLLING ...
Saving debug file (e.g., ./network_dump)
D [vsi_nn_DumpGraphNodeOutputsEx:1327]Dump 1 nodes.
I [vsi_nn_ConvertTensorToData:750]Create 43095 data.
I [vsi_nn_ConvertTensorToData:750]Create 172380 data.
I [vsi_nn_ConvertTensorToData:750]Create 689520 data.
I [vsi_nn_ConvertTensorToData:750]Create 43095 data.
 --- Top5 ---
3099: 4.750000
3098: 4.500000
1230: 3.750000
15361: 3.750000
15273: 3.500000
I [vsi_nn_ConvertTensorToData:750]Create 43095 data.
I [vsi_nn_ConvertTensorToData:750]Create 172380 data.
I [vsi_nn_ConvertTensorToData:750]Create 689520 data.
I [vsi_nn_ConvertTensorToData:750]Create 43095 data.
Pipeline is PREROLLED ...
Setting pipeline to PLAYING ...
New clock: GstSystemClock
Got EOS from element "pipeline0".
Execution ended after 0:00:00.014697750
Setting pipeline to PAUSED ...
Setting pipeline to READY ...
Setting pipeline to NULL ...
Freeing pipeline ...

```
