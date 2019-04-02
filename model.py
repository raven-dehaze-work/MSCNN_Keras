"""
MSCNN-Dehaze class
"""

import os
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.layers.merge import concatenate
from keras.optimizers import Adam,SGD
from keras.losses import mse
import numpy as np
import matplotlib.pyplot as plt



class MSCNN():
    def __init__(self,batch_size,epochs,learning_rate) -> None:
        super().__init__()
        # 模型保存地点
        self.model_dir_path = './model_save'
        self.dehazed_result_dir = './dehazed_result'
        if not os.path.exists(self.model_dir_path):
            os.mkdir(self.model_dir_path)
        if not os.path.exists(self.dehazed_result_dir):
            os.mkdir(self.dehazed_result_dir)
        self.coarse_model_path = 'coarse_net.h5'
        self.fine_model_path = 'fineModel_net.h5'

        # 设置超参数
        self.batch_size = batch_size
        self.epochs = epochs

        # 输入图片信息
        self.img_height = 240
        self.img_width = 320
        self.channel = 3

        # 建立模型
        (self.coarseModel, self.fineModel) = self.build_model()

        # 保存回调函数
        self.coarse_ckpt = ModelCheckpoint(os.path.join(self.model_dir_path,self.coarse_model_path),monitor='val_loss',
                                          verbose=1,save_best_only=True,mode='min' )
        self.fine_ckpt = ModelCheckpoint(os.path.join(self.model_dir_path, self.fine_model_path),
                                           monitor='val_loss',
                                           verbose=1, save_best_only=True, mode='min')
        # 设置优化器，损失函数等
        self.optimizer = SGD(learning_rate,0.9,0.0001)
        # self.optimizer = Adam(learning_rate)
        self.loss = mse

        self.coarseModel.compile(optimizer=self.optimizer,
                                 loss=self.loss)
        self.fineModel.compile(optimizer=self.optimizer,
                               loss=self.loss)

    def build_model(self):
        """
        建立paper中的MSCNN
        :return: corseModel和fineModel的元祖
        """
        # 通用输入
        input_img = Input((self.img_height, self.img_width, self.channel))
        coarseNet = self._build_coarseNet(input_img)
        fineNet = self._build_fineNet(input_img, coarseNet)

        # 建立coarse Model和findModel
        coarseModel = Model(inputs=input_img, outputs=coarseNet)
        fineModel = Model(inputs=input_img, outputs=fineNet)

        # summary
        coarseModel.summary()
        fineModel.summary()

        # save model to visualize
        coarseModel.save(filepath=os.path.join(self.model_dir_path, self.coarse_model_path))
        fineModel.save(filepath=os.path.join(self.model_dir_path, self.fine_model_path))

        return (coarseModel, fineModel)

    def _build_coarseNet(self, input_img):
        """
        建立coarseNet
        :param input_img: 输入图片的tensor
        :return: coarseNet
        """
        conv1 = Conv2D(5, (11, 11), padding='same', activation='relu', name='coarseNet/conv1')(input_img)
        pool1 = MaxPooling2D((2, 2), name='coarseNet/pool1')(conv1)
        upsample1 = UpSampling2D((2, 2), name='coarseNet/upsample1')(pool1)

        conv2 = Conv2D(5, (9, 9), padding='same', activation='relu', name='coarseNet/conv2')(upsample1)
        pool2 = MaxPooling2D((2, 2), name='coarseNet/pool2')(conv2)
        upsample2 = UpSampling2D((2, 2), name='coarseNet/upsample2')(pool2)

        conv3 = Conv2D(10, (7, 7), padding='same', activation='relu', name='coarseNet/conv3')(upsample2)
        pool3 = MaxPooling2D((2, 2), name='coarseNet/pool3')(conv3)
        upsample3 = UpSampling2D((2, 2), name='coarseNet/upsample3')(pool3)

        linear = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='coarseNet/linear_comb')(upsample3)
        return linear

    def _build_fineNet(self, input_img, coarseNet):
        """
        建立fineNet
        :param input_img: 输入图片的tensor
        :param coarseNet: coarseNet的Tensor
        :return: fineNet
        """
        conv1 = Conv2D(4, (7, 7), padding='same', name='fineNet/conv1')(input_img)
        pool1 = MaxPooling2D((2, 2), name='fineNet/pool1')(conv1)
        upsample1 = UpSampling2D((2, 2), name='fineNet/upsample1')(pool1)

        # 级联coarseNet
        concat = concatenate([upsample1, coarseNet], axis=3, name='concat')

        conv2 = Conv2D(5, (5, 5), padding='same', name='fineNet/conv2')(concat)
        pool2 = MaxPooling2D((2, 2), name='fineNet/pool2')(conv2)
        upsample2 = UpSampling2D((2, 2), name='fineNet/upsample2')(pool2)

        conv3 = Conv2D(10, (3, 3), padding='same', name='fineNet/conv3')(upsample2)
        pool3 = MaxPooling2D((2, 2), name='fineNet/pool3')(conv3)
        upsample3 = UpSampling2D((2, 2), name='fineNet/upsample3')(pool3)

        linear = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='fineNet/comb')(upsample3)

        return linear


    def train_on_generator(self,train_datas,val_datas):
        """
        使用生成器来训练模型--适合大数据
        :param train_datas: 字典类型的训练数据。包含训练数据总量和训练生成器。（键为：'nums'和'generator'
        :param val_datas: 字典类型的验证数据。包含验证数据总量和验证生成器
        :return: 无
        """
        train_num = train_datas['nums']
        train_generator = train_datas['generator']

        val_num = val_datas['nums']
        val_genernator = val_datas['generator']

        # 加载
        if os.path.exists(os.path.join(self.model_dir_path,self.coarse_model_path)):
            self.coarseModel.load_weights(os.path.join(self.model_dir_path,self.coarse_model_path))
        if os.path.exists(os.path.join(self.model_dir_path, self.fine_model_path)):
            self.fineModel.load_weights(os.path.join(self.model_dir_path, self.fine_model_path))

        # 开始训练
        coarse_history = self.coarseModel.fit_generator(generator=train_generator,steps_per_epoch=int(train_num/self.batch_size),epochs=self.epochs,
                                       validation_data=val_genernator,validation_steps=max(int(val_num/self.batch_size),1),
                                                        callbacks=[self.coarse_ckpt])
        fine_history = self.fineModel.fit_generator(generator=train_generator,steps_per_epoch=int(train_num/self.batch_size),epochs=self.epochs,
                                       validation_data=val_genernator,validation_steps=max(int(val_num/self.batch_size),1),
                                                    callbacks=[self.fine_ckpt])

        plt.subplot(1,2,1)
        # 绘制训练 & 验证的损失值
        plt.plot(coarse_history.history['loss'])
        plt.plot(coarse_history.history['val_loss'])
        plt.title('coarse model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')

        plt.subplot(1,2,2)
        plt.plot(fine_history.history['loss'])
        plt.plot(fine_history.history['val_loss'])
        plt.title('fine model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

    def train_on_datas(self, train_in, train_out, val_in, val_out):
        """
        使用数据训练模型--适合小数据
        :param train_in: 训练数据输入
        :param train_out: 训练数据输出
        :param val_in:  验证数据输入
        :param val_out: 验证数据输入
        :return:
        """

        samples_size = len(train_in)
        coarse_loss = 0.0
        fine_loss = 0.0
        pre_best_val_loss = np.inf
        for i in range(self.epochs):
            print('%d/%d epochs' % (i + 1, self.epochs))
            start = 0
            end = (start + self.batch_size) % samples_size
            while end <= samples_size:
                batch_train_in = train_in[start:end, :, :, :]
                batch_train_out = train_out[start:end, :, :]

                ################
                # 训练
                ################
                # coarse_loss = self.coarseModel.fit()
                coarse_loss = self.coarseModel.train_on_batch(batch_train_in, batch_train_out)
                fine_loss = self.fineModel.train_on_batch(batch_train_in, batch_train_out)

                start += self.batch_size
                end += start + self.batch_size
            # 每个epoch打印一次当前训练成果（可在这里做保存操作）
            val_fine_loss = self.fineModel.evaluate(val_in, val_out, batch_size=self.batch_size)
            print('train samples loss : %f, val samples loss %f' % (fine_loss, val_fine_loss))

            if pre_best_val_loss < val_fine_loss:
                # 保存当前模型
                self.coarseModel.save(os.path.join(self.model_dir_path,self.coarse_model_path))
                self.fineModel.save(os.path.join(self.model_dir_path,self.fine_model_path))

    def test_on_batch(self,test_in):
        """
        测试数据。
        :param test_in: 测试数据
        :return: 无
        """
        # 加载模型
        if not os.path.exists(os.path.join(self.model_dir_path,self.fine_model_path)):
            raise FileNotFoundError('model file not found, did you train model before?')
        self.fineModel.load_weights(os.path.join(self.model_dir_path,self.fine_model_path))
        test_out = self.fineModel.predict(test_in['haze']/255)

        fig = plt.figure()
        # 保存对比照片
        for idx,img in enumerate(test_out):
            # fig, axs = plt.subplots(1, 2)
            plt.subplot(1,2,1)
            plt.imshow(np.reshape(test_in['trans'][idx]/255,(self.img_height,self.img_width)),cmap='gray')
            plt.title('src')
            # axs[0,0].imshow(test_in['trans'][idx]/255)
            # axs[0,0].set_title('src')
            # axs[0,0].axis('off')

            plt.subplot(1,2,2)
            plt.imshow(np.reshape(img,(self.img_height,self.img_width)),cmap='gray')
            plt.title('trans.img')
            # plt.show()
            # axs[0, 1].imshow(img)
            # axs[0, 1].set_title('trans img')
            # axs[0, 1].axis('off')

            fig.savefig(os.path.join(self.dehazed_result_dir,'image%d.jpg'%(idx)))
