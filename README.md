<!--
 * @Author: your name
 * @Date: 2021-06-05 15:59:37
 * @LastEditTime: 2021-06-05 16:40:57
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /DeepLearningFramework/README.md
-->
# DeepLearningFramework
### 对目前主要的深度学习框架分布式学习，并对相应模块进行源码解读
#### 安装环境
* 显卡3080(两块)和2080ti(四块)
* ubuntu18.04
* cuda11.1
* cudnn8.1.0
* nvidia-docker2.x
* Nvidia-NGC nvcr.io/nvidia/tensorflow:21.02-tf2-py3/nvcr.io/nvidia/pytorch:20.06-py3
* anaconda:4.7.10
* NCCL
* horovod
* 深度学习框架（tensorflow2.4.x、pytorch1.8.0和paddllepaddle2.1.0）

#### GPU分布式训练相关概念
- [APEX](https://github.com/NVIDIA/apex) 是NVIDIA提供的一个PyTorch的扩展插件，用于帮助PyTorch实现自动混合精度以及分布式训练。
- [DALI](https://docs.nvidia.com/deeplearning/dali/release-notes/index.html) - NVIDIA Data Loading Library (DALI) 是一个高效的数据读取引擎，用于给深度学习框架加速输入数据的读取速度。
- [Horovod](https://github.com/horovod/horovod) 是一个帮助TensorFlow、PyTorch、MXNet等框架支持分布式训练的框架。
- [XLA](https://www.tensorflow.org/xla) (Accelerated Linear Algebra) 是一种深度学习编译器，可以在不改变源码的情况下进行线性代数加速
- NCCL 多机多卡的分布式训练
##### **XLA**

[XLA](https://www.tensorflow.org/xla) (Accelerated Linear Algebra)是一种深度学习编译器，可以在不改变源码的情况下进行线性代数加速。针对支持XLA的深度学习框架我们也会测试其开启或关闭状态下的性能表现。

##### **AMP 自动混合精度**

**AMP**(Automatic Mixed Precision) 自动混合精度，在GPU上可以加速训练过程，与float32精度相比，AMP在某些GPU上可以做到3倍左右的速度提升。我们对支持AMP的深度学习框架会测试其开启或关闭AMP的性能表现。
#### 深度学习框架中分布式训练API
#### tensorflow2.x版本

* 单机多卡还是多机多卡分布使用tf.distribute.Strategy和MultiWorkerMirroredStrategy 两个API 函数
#### pytorch
* pytorch分布式训练API,参考torch.distributed 

#### paddlepaddle
* PaddlePaddle Fluid
#### Oneflow
#### 借助与第三方插件和优化的分布式库

[Horovod](https://github.com/horovod/horovod)

[NCCL](https://github.com/NVIDIA/nccl) 

[DeepSpeed](https://github.com/microsoft/DeepSpeed)

### 参考文献
[lyhue1991/eat_tensorflow2_in_30_days]()

[lyhue1991/eat_pytorch_in_20_days]()

[nvidia-NGC](https://ngc.nvidia.com/catalog)

[paddlepaddle分布式训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/howto/training/single_node.html)

[新生手册」：PyTorch分布式训练](https://mp.weixin.qq.com/s?__biz=MzU4OTg3Nzc3MA==&mid=2247485399&idx=1&sn=36e50cf1dd767eb003074c84977a103d&chksm=fdc780b2cab009a45a09fa1c4f7b1ec05f84d2caed992bd28bea2d23a095fe0a6c33f24bd6d9&mpshare=1&scene=1&srcid=0331OOGyZPOwnLtdSnPl472C&sharer_sharetime=1617167185751&sharer_shareid=bb12138cbf7121360054152c6932a462&version=3.1.7.3005&platform=win#rd)

[microsoft](https://github.com/microsoft)/**[DeepSpeed](https://github.com/microsoft/DeepSpeed)**

[基于ResNet50与BERT模型的深度学习框架性能评测报告](https://github.com/Oneflow-Inc/DLPerf/blob/master/reports/dlperf_benchmark_test_report_v1_cn.md)