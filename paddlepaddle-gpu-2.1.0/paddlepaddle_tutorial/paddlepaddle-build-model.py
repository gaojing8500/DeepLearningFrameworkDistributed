# '''
# Author: your name
# Date: 2021-06-15 17:30:33
# LastEditTime: 2021-06-15 18:11:40
# LastEditors: Please set LastEditors
# Description: In User Settings Edit
# FilePath: /DeepLearningFramework/paddlepaddle-gpu-2.1.0/paddlepaddle_tutorial/paddlepaddle-build-model.py
# '''
# # '''
# # Author: your name
# # Date: 2021-06-15 17:30:33
# # LastEditTime: 2021-06-15 17:36:03
# # LastEditors: Please set LastEditors
# # Description: In User Settings Edit
# # FilePath: /DeepLearningFramework/paddlepaddle-gpu-2.1.0/paddlepaddle_tutorial/paddlepaddle-build-model.py
# # '''
# # import paddle

# # class Mnist(paddle.nn.Layer):
# #     def __init__(self):
# #         super(Mnist,self).__init__()
# #         self.flatten = paddle.nn.Flatten()
# #         self.linear_1 = paddle.nn.Linear(784,512)
# #         self.linear_2 = paddle.nn.Linear(512,10)
# #         self.relu = paddle.nn.ReLU()
# #         self.dropout = paddle.nn.Dropout(0.2)
        
# #     def forword(self,inputs):
# #         y = self.flatten(inputs)
# #         y = self.linear_1(y)
# #         y = self.linear_2(y)
# #         y = self.relu(y)
# #         y = self.dropout(y)
# #         y = self.linear_2(y)
        
# #         return y
        
# # mnist = Mnist()




# # from paddle.vision.transforms import ToTensor
# # train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
# # test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())

# # train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
# # mnist.train()

# # optimizer = paddle.optimizer.Adam(parameters=mnist.parameters())                     
# # loss=paddle.nn.CrossEntropyLoss(),
# # metrics=paddle.metric.Accuracy()


# # epochs = 2
# # for epoch in range(epochs):
# #     for batch_id, data in enumerate(train_loader()):

# #         x_data = data[0]            # 训练数据
# #         y_data = data[1]            # 训练数据标签
# #         predicts = mnist(x_data)    # 预测结果

# #         # 计算损失 等价于 prepare 中loss的设置
# #         loss = loss(predicts, y_data)

# #         # 计算准确率 等价于 prepare 中metrics的设置
# #         acc =metrics(predicts, y_data)

# #         # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中

# #         # 反向传播
# #         loss.backward()

# #         if (batch_id+1) % 900 == 0:
# #             print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id+1, loss.numpy(), acc.numpy()))

# #         # 更新参数
# #         optimizer.step()

# #         # 梯度清零
# #         optimizer.clear_grad()

# import paddle
# from paddle.vision.transforms import ToTensor

# # paddle.set_device('gpu')
# # rank = paddle.distributed.get_rank()
# # if paddle.distributed.get_world_size() > 1:
# #         paddle.distributed.init_parallel_env()

# # 加载数据集
# train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
# test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())

# # 定义网络结构
# mnist = paddle.nn.Sequential(
#     paddle.nn.Flatten(1, -1),
#     paddle.nn.Linear(784, 512),
#     paddle.nn.ReLU(),
#     paddle.nn.Dropout(0.2),
#     paddle.nn.Linear(512, 10)
# )

# model = paddle.Model(mnist)

# # model = paddle.DataParallel(model)

# # 为模型训练做准备，设置优化器，损失函数和精度计算方式
# model.prepare(optimizer=paddle.optimizer.Adam(parameters=model.parameters()),
#               loss=paddle.nn.CrossEntropyLoss(),
#               metrics=paddle.metric.Accuracy())

# model.fit(train_dataset,
#           epochs=5,
#           batch_size=64,
#           verbose=1)


import paddle
# 第1处改动 导入分布式训练所需的包
import paddle.distributed as dist
from paddle.vision.transforms import ToTensor

# 加载数据集
train_dataset = paddle.vision.datasets.MNIST(mode='train',transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test',transform=ToTensor())

# 第2处改动，初始化并行环境
dist.init_parallel_env()

# 定义网络结构
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(1, -1),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)

# 用 DataLoader 实现数据加载
train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 第3处改动，增加paddle.DataParallel封装
mnist = paddle.DataParallel(mnist)
mnist.train()

# 设置迭代次数
epochs = 5

# 设置优化器
optim = paddle.optimizer.Adam(parameters=mnist.parameters())

for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):

        x_data = data[0]            # 训练数据
        y_data = data[1]            # 训练数据标签
        predicts = mnist(x_data)    # 预测结果

        # 计算损失 等价于 prepare 中loss的设置
        loss = paddle.nn.functional.cross_entropy(predicts, y_data)

        # 计算准确率 等价于 prepare 中metrics的设置
        acc = paddle.metric.accuracy(predicts, y_data)

        # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中

        # 反向传播
        loss.backward()

        if (batch_id+1) % 1800 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))

        # 更新参数
        optim.step()

        # 梯度清零
        optim.clear_grad()