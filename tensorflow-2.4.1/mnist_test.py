'''
Author: your name
Date: 2021-05-24 14:35:21
LastEditTime: 2021-05-24 14:36:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PTM/tensorflow-2.3.1/mnist_test.py
'''
##简答测试官方的手写字模型

from numpy.lib.function_base import gradient
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D
from tensorflow.keras import Model, optimizers

'''
可能需要对gpu资源的分配
ensorflow.python.framework.errors_impl.UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
[[node sequential/conv2d/Conv2D (defined at tf_keras_classification_model-cnn.py:107) ]] [Op:__inference_distributed_function_930]
Function call stack:
distributed_function
————————————————
版权声明：本文为CSDN博主「TFATS」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/TFATS/article/details/113978075
'''
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# # 定义TensorFlow配置
# config = ConfigProto()
# # 配置GPU内存分配方式，按需增长，很关键
# config.gpu_options.allow_growth = True
# # 在创建session的时候把config作为参数传进去
# session = InteractiveSession(config=config)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


class MiniDataLoad():
    def __init__(self):
        self.minst = tf.keras.datasets.fashion_mnist
    def minst_dataset_process(self):
        (x_train, y_train), (x_test, y_test) = self.minst.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        train_ds = tf.data.Dataset.from_tensor_slices(
                            (x_train, y_train)).shuffle(10000).batch(32* strategy.num_replicas_in_sync)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32* strategy.num_replicas_in_sync)

        return train_ds,test_ds

class LossFuction():
    def __init__(self):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


##tf2.x在构建模型上面要易用性比tf1.x要好很多
class MiniModel(Model):
    def __init__(self):
        super(MiniModel,self).__init__()
        self.conv1 = Conv2D(32,3,activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128,activation="relu")
        self.d2 = Dense(10,activation="softmax")

    def call(self,x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

class TrainStep():
    def __init__(self,lossfuction):
        self.name = "train step"
        self.lossfuction = lossfuction

    def distrbuted_strategy(self):
        with strategy.scope():
            self.model = MiniModel()

    ##采用静态图进行训练
    @tf.function
    def train_step(self,images,labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.lossfuction.loss_object(labels,predictions)
        gradients = tape.gradient(loss,self.model.trainable_variables)
        self.lossfuction.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        self.lossfuction.train_loss(loss)
        self.lossfuction.train_accuracy(labels,predictions)
    @tf.function
    def test_step(self,images, labels):
        predictions = self.model(images)
        t_loss = self.lossfuction.loss_object(labels, predictions)

        self.lossfuction.test_loss(t_loss)
        self.lossfuction.test_accuracy(labels, predictions)

def main():
    EPOCHS = 10
    lossfuction = LossFuction()
    trainStep = TrainStep(lossfuction)
    mnist = MiniDataLoad()
    train_ds,test_ds = mnist.minst_dataset_process()
    trainStep.distrbuted_strategy()

    for epoch in range(EPOCHS):
        
    # 在下一个epoch开始时，重置评估指标
        lossfuction.train_loss.reset_states()
        lossfuction.train_accuracy.reset_states()
        lossfuction.test_loss.reset_states()
        lossfuction.test_accuracy.reset_states()

        for images, labels in train_ds:
            trainStep.train_step(images, labels)

        for test_images, test_labels in test_ds:
            trainStep.test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print (template.format(epoch+1,
                         lossfuction.train_loss.result(),
                         lossfuction.train_accuracy.result()*100,
                         lossfuction.test_loss.result(),
                         lossfuction.test_accuracy.result()*100))
if __name__ == "__main__":
    main()



        