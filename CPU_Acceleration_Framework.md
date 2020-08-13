# Model-Acceleration-Framework
**Purpose**: 使得手机端cpu的速度快，开发者能够将深度学习算法轻松移植到手机端高效执行。


# ONNX is standard format of model
Open Neural Network Exchange (ONNX) is an open standard format for representing machine learning models. ONNX is supported by a community of partners who have implemented it in many frameworks and tools.





## Tencent  -- NCNN & TNN
### 1. NCNN

a. 无第三方库的依赖，不依赖BLAS/NNPACK等计算框架 
    BLAS线性代数库（矢量的线性组合， 矩阵乘以矢量， 矩阵乘以矩阵等）
    NNPACK由facebook开发，是一个加速神经网络计算的加速包，NNPACK可以在多核CPU平台上提高卷积层计算性能。
b. ARM NEON 汇编级良心优化，计算速度极快
c. 精细的内存管理和数据结构设计，内存占用极低
d. 支持基于全新低消耗的 vulkan api GPU 加速

e. 可扩展的模型设计，支持 8bit 量化和半精度浮点存储，可导入 caffe/pytorch/mxnet/onnx/darknet 模型
f. 支持直接内存零拷贝引用加载网络模型
g. 可注册自定义层实现并扩展

|            | Windows | Linux | Android | MacOS | iOS  |
| ---------- | ------- | ----- | ------- | ----- | ---- |
| intel-cpu  | ✔️       | ✔️     | ❔       | ✔️     | /    |
| intel-gpu  | ✔️       | ✔️     | ❔       | ❔     | /    |
| amd-cpu    | ✔️       | ✔️     | ❔       | ✔️     | /    |
| amd-gpu    | ✔️       | ✔️     | ❔       | ❔     | /    |
| nvidia-gpu | ✔️       | ✔️     | ❔       | ❔     | /    |
| qcom-cpu   | ❔       | ✔️     | ✅       | /     | /    |
| qcom-gpu   | ❔       | ✔️     | ✔️       | /     | /    |
| arm-cpu    | ❔       | ❔     | ✅       | /     | /    |
| arm-gpu    | ❔       | ❔     | ✔️       | /     | /    |
| apple-cpu  | /       | /     | /       | /     | ✅    |
| apple-gpu  | /       | /     | /       | /     | ✔️    |

### 2.TNN

轻量级推理框架，同时拥有跨平台、高性能、模型压缩、代码裁剪,

目前 TNN 目前仅支持 **CNN** 等常用网络结构，RNN、GAN等网络结构正在逐步开发中。

- 计算优化
  - 针对不同架构在硬件指令发射、吞吐、延迟、缓存带宽、缓存延迟、寄存器数量等特点，深度优化底层算子，极致利用硬件算力
  - 主流硬件平台(CPU: ARMv7， ARMv8， GPU: Mali， Adreno， Apple) 深度调优
  - CNN 核心卷积运算通过 Winograd，Tile-GEMM， Direct Conv 等多种算法实现，保证不同参数、计算尺度下高效计算
  - Op 融合：离线分析网络计算图，多个小 Op（计算量小、功能较简单）融合运算，减少反复内存读取、kernel 启动等开销
- 低精度优化
  - 支持 INT8， FP16 低精度计算，减少模型大小、内存消耗，同时利用硬件低精度计算指令加速计算
  - 支持 INT8 Winograd 算法，(输入6bit)， 在精度满足要求的情况下，进一步降低模型计算复杂度
  - 支持单模型多种精度混合计算，加速计算同时保证模型精度
- 内存优化
  - 高效”内存池”实现：通过 DAG 网络计算图分析，实现无计算依赖的节点间复用内存，降低 90% 内存资源消耗
  - 跨模型内存复用：支持外部实时指定用于网络内存，实现“多个模型，单份内存”。

![img](https://camo.githubusercontent.com/899e5734136182e76bdc412c35c52cab5d17aca6/68747470733a2f2f67697465652e636f6d2f64617272656e33642f746e6e2d7265736f757263652f7261772f6d61737465722f646f632f636e2f696d67732f746e6e5f6172636869746563742e6a7067)



| device | support |
| ------ | ------- |
| ARMv7  | Yes     |
| ARMv8  | Yes     |
| OpenCL | Yes     |
| Metal  | Yes     |

如果使用TNN，则需要将训练好的模型 先转换成ONNX，在转换成TNN模型.

| 原始模型   | 转换工具        | 目标模型 | 转换工具 | TNN  |
| ---------- | --------------- | -------- | -------- | ---- |
| PyTorch    | pytorch export  | ONNX     | onnx2tnn | TNN  |
| TensorFlow | tensorflow-onnx | ONNX     | onnx2tnn | TNN  |
| Caffe      | caffe2onnx      | ONNX     | onnx2tnn | TNN  |

### 模型转换详细介绍

convert2tnn 只是对多种模型转换的工具的封装，根据第一部分 “模型转换介绍”中原理说明，你也可以先将原始模型转换成 ONNX，然后再将 ONNX 模型转换成 TNN 模型。我们提供了如何手动的将 Caffe、PyTorch、TensorFlow 模型转换成 ONNX 模型，然后再将 ONNX 模型转换成 TNN 模型的文档。如果你在使用 convert2tnn 转换工具遇到问题时，我们建议你了解下相关的内容，这有可能帮助你更加顺利的进行模型转换。

- [onnx2tnn](https://github.com/Tencent/TNN/blob/master/doc/cn/user/onnx2tnn.md)
- [pytorch2tnn](https://github.com/Tencent/TNN/blob/master/doc/cn/user/onnx2tnn.md)
- [tf2tnn](https://github.com/Tencent/TNN/blob/master/doc/cn/user/tf2tnn.md)
- [caffe2tnn](https://github.com/Tencent/TNN/blob/master/doc/cn/user/caffe2tnn.md)

## XiaoMi -- MACE

MACE (Mobile AI Compute Engine)

一个专为移动端异构计算平台(支持Android, iOS, Linux, Windows)优化的神经网络计算框架

- 性能
  - 代码经过NEON指令，OpenCL以及Hexagon HVX专门优化，并且采用 [Winograd算法](https://arxiv.org/abs/1509.09308)来进行卷积操作的加速。 此外，还对启动速度进行了专门的优化。
- 功耗
  - 支持芯片的功耗管理，例如ARM的big.LITTLE调度，以及高通Adreno GPU功耗选项。
- 系统响应
  - 支持自动拆解长时间的OpenCL计算任务，来保证UI渲染任务能够做到较好的抢占调度， 从而保证系统UI的相应和用户体验。
- 内存占用
  - 通过运用内存依赖分析技术，以及内存复用，减少内存的占用。另外，保持尽量少的外部 依赖，保证代码尺寸精简。
- 模型加密与保护
  - 模型保护是重要设计目标之一。支持将模型转换成C++代码，以及关键常量字符混淆，增加逆向的难度。
- 硬件支持范围
  - 支持高通，联发科，以及松果等系列芯片的CPU，GPU与DSP(目前仅支持Hexagon)计算加速。CPU模式支持Android, iOS, Linux等系统。
- 模型格式支持
  - 支持[TensorFlow](https://github.com/tensorflow/tensorflow)， [Caffe](https://github.com/BVLC/caffe)和[ONNX](https://github.com/onnx/onnx)等模型格式。

## Alibaba -- MNN

包括在卷积和反卷积中应用Winograd算法、在矩阵乘法中应用Strassen算法、低精度计算、Neon优化、手写汇编、多线程优化、内存复用、异构计算等。

### 高性能

- 不依赖任何第三方计算库，依靠大量手写汇编实现核心运算，充分发挥ARM CPU的算力。
- iOS设备上可以开启GPU加速（Metal），常用模型上快于苹果原生的CoreML。
- Android上提供了`OpenCL`、`Vulkan`、`OpenGL`三套方案，尽可能多地满足设备需求，针对主流GPU（`Adreno`和`Mali`）做了深度调优。
- 卷积、转置卷积算法高效稳定，对于任意形状的卷积均能高效运行，广泛运用了 Winograd 卷积算法，对3x3 -> 7x7之类的对称卷积有高效的实现。
- 针对ARM v8.2的新架构额外作了优化，新设备可利用FP16半精度计算的特性获得两倍提速。

### 轻量性

- 针对端侧设备特点深度定制和裁剪，无任何依赖，可以方便地部署到移动设备和各种嵌入式设备中。
- iOS平台：armv7+arm64静态库大小5MB左右，链接生成可执行文件增加大小620KB左右，metallib文件600KB左右。
- Android平台：so大小400KB左右，OpenCL库400KB左右，Vulkan库400KB左右。

### 通用性

- 支持`Tensorflow`、`Caffe`、`ONNX`等主流模型文件格式，支持`CNN`、`RNN`、`GAN`等常用网络。
- 转换器支持149个`Tensorflow`OP、58个`TFLite` OP、47个`Caffe` OP、74个`ONNX` OP；各计算设备后端支持的MNN OP数：CPU 111个，ARM V8.2 6个，Metal 55个，OpenCL 43个，Vulkan 32个。
- 支持iOS 8.0+、Android 4.3+和具有POSIX接口的嵌入式设备。
- 支持异构设备混合计算，目前支持CPU和GPU。

### 易用性

- 有高效的图像处理模块，覆盖常见的形变、转换等需求，一般情况下，无需额外引入libyuv或opencv库处理图像。
- 支持回调机制，可以在网络运行中插入回调，提取数据或者控制运行走向。
- 支持只运行网络中的一部分，或者指定CPU和GPU间并行运行。
- （BETA）MNN Python API，让算法工程师可以轻松地使用MNN构图、训练、量化训练，无需编写C++。

![architecture](https://github.com/alibaba/MNN/raw/master/doc/architecture.png)

*Vulkan是*一个跨平台的2D和3D绘图应用程序接口（API）

## DNNLibrary

DNNLibrary is a wrapper of NNAPI ("DNNLibrary" is for "**d**aquexian's **NN**API library). It lets you easily make the use of the new NNAPI introduced in Android 8.1. You can convert your onnx model into `daq` and run the model directly.

**ONNX model -->  onnx2daq  --> daq model** 

a convert tool like [MMdnn](https://github.com/Microsoft/MMdnn) to convert the caffe model to the ONNX model, then convert it to `daq` using `onnx2daq`.

Tensorflow lite is support NNAPI, But **dilated convolution** which is widely used in segmentation and **prelu** are not supported.

| _                      | TF Lite                                                    | DNNLibrary       |
| ---------------------- | ---------------------------------------------------------- | ---------------- |
| Supported Model Format | TensorFlow                                                 | ONNX             |
| Dilated Convolution    | ❌                                                          | ✔️                |
| Ease of Use            | ❌ (Bazel build system, not friendly to Android developers) | ✔️                |
| Quantization           | ✔️                                                          | ✔️ (since 0.6.10) |



![DNNLibrary-rk3399.png](https://github.com/JDAI-CV/DNNLibrary/blob/master/images/DNNLibrary-rk3399.png?raw=true)

![Benchmark on OnePlus 6T](https://github.com/JDAI-CV/DNNLibrary/raw/master/images/DNNLibrary-oneplus6t.png)

![Benchmark on Huawei Honor V10](https://github.com/JDAI-CV/DNNLibrary/raw/master/images/DNNLibrary-huaweihonorv10.png)

## Apple CoreML

