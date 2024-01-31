    #!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2021/11/8
# @Author: zhou
# @File  : 交叉验证并绘制ROC曲线和混淆矩阵
import warnings
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib as mpl
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, precision_score, recall_score,f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier as KNN
import pandas as pd
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
warnings.filterwarnings(action='ignore')
'''
代码包含三部分
第一部分是交叉验证封装函数，内部需要根据输入的模型修改对应的数据处理部分，我都写好了注释，用的哪一个模型就对应取消哪部分注释
第二部分是各个网络模型的结构定义，需要修改哪个模型就跳到那一部分修改，也进行了一个函数封装，需要使用哪个函数就输入模型名称会自动返回该model
第三部分是主函数，读入数据，分割数据，调用函数等等
如果不需要改变网络结构的话可f 以直接跳到最后主函数部分
'''
def show_history(history): # 显示训练过程中的学习曲线
    loss = history.history['loss'] #训练损失
    val_loss = history.history['val_loss'] #验证损失
    epochs = range(1, len(loss) + 1) #训练轮次
    plt.figure(figsize=(12,4)) # 图片大小
    plt.subplot(1, 2, 1) #子图1
    plt.plot(epochs, loss, 'r', label='Training loss') #训练损失
    plt.plot(epochs, val_loss, 'b', label='Validation loss') #验证损失
    plt.title('Training and validation loss') #图题
    plt.xlabel('Epochs') #X轴文字
    plt.ylabel('Loss') #Y轴文字
    plt.legend() #图例
    acc = history.history['acc'] #训练准确率
    val_acc = history.history['val_acc'] #验证准确率
    plt.subplot(1, 2, 2) #子图2
    plt.plot(epochs, acc, 'r', label='Training acc') #训练准确率
    plt.plot(epochs, val_acc, 'b', label='Validation acc') #验证准确率
    plt.title('Training and validation accuracy') #图题
    plt.xlabel('Epochs') #X轴文字
    plt.ylabel('Accuracy') #Y轴文字
    plt.legend() #图例
    plt.show() #绘图
    # plt.savefig(
    #     r'C:\Users\27152\Desktop\fanxiu\loss\\' + str(
    #     1) + '.jpg', dpi=300, bbox_inches="tight", pad_inches=0.0,transparent=True)

def cross_validation(df_train, labels_train, df_test, labels_test, k, model):
    '''
    df_train, labels_train:训练集、训练集标签
    df_test, labels_test:测试集、测试集标签
    k：交叉验证次数
    model：验证的模型
    '''
    # 用来保存每次预测的概率值
    test_predict = []

    # 用来保存每次预测的精确率
    test_precision = []

    # 用来保存每次预测的召回率
    test_recall = []


    # 用来保存每次预测的F1分数
    test_F1_Score = []

    # n_splits:交叉验证的个数
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    # 记录验证次数
    cnt = 0
    # 记录每折的准确率
    allacc1 = 0
    allacc2 = 0
    allacc3 = 0
    #测试集
    label_test = labels_test
    labels_test = np_utils.to_categorical(labels_test, num_classes=2)
    #AleNet、ResNet、GoogleNet、twoLSTM
    # df_test = df_test.reshape(-1, a, 1)
    # CNN、RNN
    df_test = df_test.reshape(-1, 1, a)

    print("五折交叉验证的准确率如下：")
    print(np.array(df_train).shape,np.array(labels_train).shape)
    # 利用模型划分数据集和目标变量 为一一对应的下标
    for i, (train, test) in enumerate(cv.split(df_train, labels_train)):#需要拆分y_trian
        # 记录训练次数
        global count
        count += 1
        cnt += 1
        print('第%d次迭代: ' % cnt)

        #训练集
        label_train = labels_train[train]
        y_train = np_utils.to_categorical(labels_train[train], num_classes=2)
        #AleNet、ResNet、GoogleNet、twoLSTM
        # data_train = df_train[train].reshape(-1,a,1)
        #CNN、RNN
        data_train = df_train[train].reshape(-1,1,a)
        #MCNN、ANN
        # data_train = df_train[train]

        #验证集
        label_val = labels_train[test]
        labels_val = np_utils.to_categorical(labels_train[test], num_classes=2)
        #AleNet、ResNet、GoogleNet、twoLSTM
        # df_val = df_train[test].reshape(-1,a,1)
        #CNN、RNN
        df_val = df_train[test].reshape(-1,1, a)
        # MCNN、ANN
        # df_val = df_train[test]
        #训练模型
        # filepath = r'./best_model' + str(count) + '.h5'
        # checkpoint = ModelCheckpoint(
        #                              # filepath,  # (就是你准备存放最好模型的地方),
        #                              monitor='val_acc',  # (或者换成你想监视的值,比如acc,loss, val_loss,其他值应该也可以,还没有试),
        #                              verbose=1,  # (如果你喜欢进度条,那就选1,如果喜欢清爽的就选0,verbose=冗余的),
        #                              # save_best_only='True',  # (只保存最好的模型,也可以都保存),
        #                              # save_weights_only='True',  # 只保存权重
        #                              mode='max',
        #                              # (如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
        #                              period=1
        # )  # (checkpoints之间间隔的epoch数)
        history = model.fit(data_train, y_train, batch_size=30, epochs=100, verbose=2, shuffle=True,
                            validation_data=(df_val, labels_val), )
        # model.fit(data_train, y_train, batch_size=32, validation_data=(df_val, labels_val), epochs=10)  # 批处理个数32
        show_history(history)
        # 训练集准确率
        test_predict1 = model.predict(data_train)
        test_predict1 = np.argmax(test_predict1, axis=1)
        accuracy1 = accuracy_score(label_train, test_predict1)
        print('训练集准确率: ', accuracy1)
        allacc1 = allacc1 + accuracy1

        # 验证集准确率
        test_predict2 = model.predict(df_val)
        test_predict2 = np.argmax(test_predict2, axis=1)
        accuracy2 = accuracy_score(label_val, test_predict2)
        print('验证集准确率: ', accuracy2)
        allacc2 = allacc2 + accuracy2

        # 测试集准确率
        test_predict3 = model.predict(df_test)
        test_predict3 = np.argmax(test_predict3, axis=1)
        accuracy3 = accuracy_score(label_test, test_predict3)
        print('测试集准确率: ', accuracy3)
        allacc3 = allacc3 + accuracy3


        # 记录每次测试的概率值
        predict_p = model.predict(df_test)
        test_predict.append(predict_p)

        # 计算精确率
        accuracy4 = precision_score(label_test, test_predict3, average=None)
        test_precision.append(accuracy4)

        # 计算召回率
        accuracy5 = recall_score(label_test, test_predict3, average=None)
        test_recall.append(accuracy5)
        # 计算F1分数
        accuracy6 = f1_score(label_test, test_predict3, average=None)
        # print(accuracy6)
        print('测试集F1-Score: ', accuracy6)
        test_F1_Score.append(accuracy6)

    # 计算十次平均预测概率
    mean_predict = test_predict[0]
    for i in range(1, 5):
        mean_predict += test_predict[i]
    mean_predict = mean_predict / 5

    # 根据平均概率求出最终的标签
    predict_idx = np.argmax(mean_predict, axis=1)

    # 计算十次平均精确率
    mean_precision = test_precision[0]
    for i in range(1, 5):
        mean_precision += test_precision[i]
    mean_precision = mean_precision / 5

    # 计算十次平均召回率
    mean_recall = test_recall[0]
    for i in range(1, 5):
        mean_recall += test_recall[i]
    mean_recall = mean_recall / 5

    # 计算十次平均F1分数
    mean_F1_Score = test_F1_Score[0]
    for i in range(1, 5):
        mean_F1_Score += test_F1_Score[i]
    mean_F1_Score = mean_F1_Score / 5

    print('=========五折交叉验证之后最终的结果==========')
    print('训练集的平均准确率: ', allacc1 / 5)
    print('验证集的平均准确率: ', allacc2 / 5)
    print('测试集的平均准确率: ', allacc3 / 5)

    print('=======五折交叉验证之后测试集的测试指标=======')
    print('样本类别:\t\t\t\t0\t\t\t1\t\t\t')
    print('测试集的平均精确率: ', mean_precision)
    print('测试集的平均召唤率: ', mean_recall)
    print('测试集的平均F1分数: ', mean_F1_Score)


    # 绘制ROC图
    # 设置种类
    n_classes = 2

    # 分类模型的decision_function返回结果的形状与样本数量相同，且返回结果的数值表示模型预测样本属于positive正样本的可信度。
    y_score = mean_predict

    # 计算每一类的ROC
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 17,
             }
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 将数据转化为矩阵
    # y_test = np.array(labels_test)
    # # print(y_test)
    # y_test = label_binarize(y_test, classes=[0, 1])
    # y_test = np_utils.to_categorical(y_test, num_classes=2)  # 将标签转化为形如(样本数, 类别数)的二值序列。
    # print(y_test)
    for i in range(n_classes):
        # [:, i]取i列的所有行，
        fpr[i], tpr[i], _ = roc_curve(labels_test[:, i], y_score[:, i])
        # ROC曲线下的面积大小
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微观平均ROC曲线和ROC面积
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # # 1D-CNN
    # Note = open('1D-CNN_fpr.txt', mode='w')
    # for i in fpr["micro"]:
    #     Note.write(str(i) + '\n')
    # Note.close()
    # Note = open('1D-CNN_tpr.txt', mode='w')
    # for i in tpr["micro"]:
    #     Note.write(str(i) + '\n')
    # Note.close()
    # MFCC-1D-CNN
    Note = open('MFCC-1D-CNN_fpr.txt', mode='w')
    for i in fpr["micro"]:
        Note.write(str(i) + '\n')
    Note.close()
    Note = open('MFCC-1D-CNN_tpr.txt', mode='w')
    for i in tpr["micro"]:
        Note.write(str(i) + '\n')
    Note.close()
    # # PLS-1D-CNN
    # Note = open('PLS-1D-CNN_fpr.txt', mode='w')
    # for i in fpr["micro"]:
    #     Note.write(str(i) + '\n')
    # Note.close()
    # Note = open('PLS-1D-CNN_tpr.txt', mode='w')
    # for i in tpr["micro"]:
    #     Note.write(str(i) + '\n')
    # Note.close()

    # 计算宏观平均ROC曲线和ROC面积
    # 一、汇总所有假阳性率
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # 二、在此点对所有ROC曲线进行插值
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # 三、最后求平均值并计算AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'black'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('False Positive Rate', font2)
    plt.ylabel('True Positive Rate', font2)
    plt.title('1D-CNN', font2)
    plt.legend(loc="lower right")
    plt.show()


    # 绘制混淆矩阵
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 17,
             }

    # 规定使用黑体
    mpl.rcParams["font.family"] = "SimHei"
    # 规定使用黑体
    mpl.rcParams["font.size"] = "17"
    # 用来正常显示符号
    mpl.rcParams["axes.unicode_minus"] = False

    # 输出混淆矩阵的值。labels指定预测的标签，前面的为正例，后面的为负例。
    y_test = np.argmax(labels_test, axis=1)
    print("混淆矩阵值如下：")
    print(confusion_matrix(y_true=label_test, y_pred=predict_idx, labels=[0, 1]))
    # matrix就是二维ndarray数组类型。
    matrix = confusion_matrix(y_true=label_test, y_pred=predict_idx, labels=[0, 1])

    # 混淆矩阵图。
    plt.matshow(matrix, cmap=plt.cm.Blues, alpha=0.5)
    # 依次遍历矩阵的行与列。
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # va：指定垂直对齐方式。
            # ha：指定水平对齐方式。
            plt.text(x=j, y=i, s=matrix[i, j], va='center', ha='center')
    plt.xlabel('Predicted Condition', font2)
    plt.ylabel('True Condition', font2)
    plt.title('1D-CNN', font2)
    plt.show()



def ModelSelect(name):
#============================================AlexNET=============================================================#
    if name == 'AlexNet':
        from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Reshape, Embedding, GRU, \
            UpSampling1D, \
            ZeroPadding1D, Activation, Dropout, BatchNormalization, concatenate, LeakyReLU
        from tensorflow.keras.layers import Dense  # dense 全连接层
        import tensorflow.keras
        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_curve, auc
        from tensorflow.keras.models import Sequential  # 序列模型各层之间是依次顺序的线性关系
        # from tensorflow.keras.utils import np_utils  # 神经网络可视化
        from tensorflow.python.keras.utils import np_utils
        import matplotlib.pyplot as plt
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ModelCheckpoint  # 保存最好的模型
        #  通过.add将网络层堆叠，形成模型
        modelAlexNet = Sequential()
        modelAlexNet.add(
            Conv1D(24, 11, strides=4, input_shape=(a, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
        # modelAlexNET.add(MaxPooling1D(pool_size=2,strides=2))
        modelAlexNet.add(Conv1D(64, 5, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
        # modelAlexNET.add(MaxPooling1D(pool_size=2,strides=2))
        modelAlexNet.add(Conv1D(128, 3, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
        modelAlexNet.add(Conv1D(64, 3, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
        modelAlexNet.add(Conv1D(64, 3, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
        modelAlexNet.add(MaxPooling1D(pool_size=2, strides=2))
        modelAlexNet.add(Flatten())
        modelAlexNet.add(Dense(128, activation='relu'))
        modelAlexNet.add(Dropout(0.5))
        modelAlexNet.add(Dense(128, activation='relu'))
        modelAlexNet.add(Dropout(0.5))
        modelAlexNet.add(Dense(2, activation='softmax'))

        # adam = Adam(lr=0.0001, beta_1=0.9, beta_2=70.999, epsilon=1e-07, decay=3e-6)
        # .compile 编译模型 （损失函数 优化器）
        modelAlexNet.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        modelAlexNet.summary()  # 通过model.summary()输出模型各层的参数状况
#===============================================ResNet==================================================================#
    if name == 'ResNet':
        from sklearn.model_selection import train_test_split
        from tensorflow.keras import Input
        from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Reshape, Embedding, GRU, \
            UpSampling1D, \
            ZeroPadding1D, Activation, Dropout, BatchNormalization, concatenate, LeakyReLU
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam
        import tensorflow.keras
        from tensorflow.keras.models import Model
        from sklearn.metrics import roc_curve, auc
        import numpy as np
        from tensorflow.python.keras.utils import np_utils
        import matplotlib.pyplot as plt
        adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                    decay=3e-8)  # alpha：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）
        from tensorflow.keras.layers import add
        import pandas as pd
        from tensorflow.keras.callbacks import ModelCheckpoint
        def Conv1d_BN(x, nb_filter, kernel_size, strides=1, padding='same', name=None):
            if name is not None:
                bn_name = name + '_bn'
                conv_name = name + '_conv'
            else:
                bn_name = None
                conv_name = None

            x = Conv1D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
            x = BatchNormalization(axis=2, name=bn_name)(x)
            return x


        def Conv_Block(inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
            x = Conv1d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
            x = Conv1d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
            if with_conv_shortcut:
                shortcut = Conv1d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
                x = add([x, shortcut])
                return x
            else:
                x = add([x, inpt])
                return x


        inpt = Input(shape=(a, 1))
        x = ZeroPadding1D(3)(inpt)
        x = Conv1d_BN(x, nb_filter=24, kernel_size=7, strides=2, padding='valid')
        # x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
        # (56,56,64)
        x = Conv_Block(x, nb_filter=24, kernel_size=3)
        x = Conv_Block(x, nb_filter=24, kernel_size=3)
        # x = Conv_Block(x,nb_filter=64,kernel_size=3)
        # (28,28,128)
        x = Conv_Block(x, nb_filter=48, kernel_size=3, strides=2, with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=48, kernel_size=3)
        # x = Conv_Block(x,nb_filter=32,kernel_size=3)
        # x = Conv_Block(x,nb_filter=128,kernel_size=3)
        # (14,14,256)
        x = Conv_Block(x, nb_filter=64, kernel_size=3, strides=2, with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=64, kernel_size=3)
        # x = Conv_Block(x,nb_filter=48,kernel_size=3)
        # x = Conv_Block(x,nb_filter=256,kernel_size=3)
        # x = Conv_Block(x,nb_filter=256,kernel_size=3)
        # x = Conv_Block(x,nb_filter=256,kernel_size=3)
        # (7,7,512)
        x = Conv_Block(x, nb_filter=128, kernel_size=3, strides=2, with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=128, kernel_size=3)
        # x = Conv_Block(x,nb_filter=32,kernel_size=3)
        # x = AveragePooling1D(pool_size=5)(x)
        x = Flatten()(x)
        x = Dense(2, activation='softmax')(x)

        modelResNet = Model(inputs=inpt, outputs=x)

        modelResNet.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
            'accuracy'])  # 目标函数由mse、mae、mape、msle、squared_hinge、hinge、binary_crossentropy、categorical_crossentrop、sparse_categorical_crossentrop等
        modelResNet.summary()  # 通过model.summary()输出模型各层的参数状况

        from tensorflow.keras.callbacks import ModelCheckpoint
        # filepath = r'./MC/best_model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        lrreduce = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0,
                                                                mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)  # 回调函数
#==================================================MCNN================================================================#
    if name == 'MCNN':
        import pandas as pd
        from keras import Input
        from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Reshape, Embedding, GRU, UpSampling1D, \
            ZeroPadding1D, Activation, Dropout, BatchNormalization, concatenate, LeakyReLU
        from keras.layers import Dense
        from keras.optimizers import adam_v2
        import keras
        from keras.models import Model
        import numpy as np
        from keras.utils import np_utils
        import matplotlib.pyplot as plt
        from tensorflow.keras.callbacks import ModelCheckpoint
        import os
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.layers import add
        from sklearn.model_selection import KFold
        from sklearn.svm import LinearSVC
        from sklearn.feature_selection import SelectFromModel
        # Input():用来实例化一个keras张量
        inpt = Input(shape=(a,))
        # 转置
        x = Reshape((-1, 1))(inpt)
        # 批归一化 training=False/0是默认的
        x = BatchNormalization()(x, training=False)
        # random_normal：正态分布初始化；
        x1 = Conv1D(filters=16, kernel_size=4, strides=1, padding='same', kernel_initializer='random_normal',
                    bias_initializer='zeros')(x)
        x1 = BatchNormalization()(x1, training=False)
        x1 = LeakyReLU(alpha=0.1)(x1)

        x2 = Conv1D(filters=16, kernel_size=8, strides=1, padding='same', kernel_initializer='random_normal')(x1)
        x2 = BatchNormalization()(x2, training=False)
        x2 = LeakyReLU(alpha=0.1)(x2)

        x3 = Conv1D(filters=16, kernel_size=16, strides=1, padding='same', kernel_initializer='random_normal')(x2)
        x3 = BatchNormalization()(x3, training=False)
        x3 = LeakyReLU(alpha=0.1)(x3)

        x = concatenate([x1, x2, x3])
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, kernel_initializer='random_normal')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation='softmax', kernel_initializer='random_normal')(x)

        modelMCNN = Model(inputs=[inpt], outputs=[x])

        modelMCNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
            'accuracy'])  # 目标函数由mse、mae、mape、msle、squared_hinge、hinge、binary_crossentropy、categorical_crossentrop、sparse_categorical_crossentrop等
        modelMCNN.summary()  # 通过model.summary()输出模型各层的参数状况
#==================================================GoogleNet=============================================================#
    if name == 'GoogleNet':
        import tensorflow as tf
        import os
        from sklearn.model_selection import train_test_split
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
        config.gpu_options.allow_growth = True  # allocate dynamically
        from tensorflow.keras.utils import plot_model
        import matplotlib.pyplot as plt
        plt.style.use('seaborn')
        import seaborn as sns
        sns.set_style("whitegrid")
        from sklearn.metrics import roc_curve, auc  ###计算roc和auc
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Reshape, Embedding, GRU, \
            UpSampling1D, \
            ZeroPadding1D, Activation, Dropout, BatchNormalization, concatenate, LeakyReLU, Lambda
        from tensorflow.keras import backend as B
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.regularizers import l2
        import numpy as np
        from tensorflow.keras import Input
        from tensorflow.keras.models import Model
        from tensorflow.python.keras.utils import np_utils
        import matplotlib.pyplot as plt
        # from keras.optimizers import SGD
        from tensorflow.keras.optimizers import Adam
        import pandas as pd
        from tensorflow.keras.callbacks import ModelCheckpoint
        def Conv1d_BN(x, nb_filter, kernel_size, padding='same', strides=1, name=None):
            if name is not None:
                bn_name = name + '_bn'
                conv_name = name + '_conv'
            else:
                bn_name = None
                conv_name = None

            x = Conv1D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
            x = BatchNormalization(axis=2, name=bn_name)(x)

            return x


        # CNN_inception块
        def Inception(x, nb_filter):
            branch1x1 = Conv1d_BN(x, nb_filter, 1, padding='same', strides=1, name=None)

            branch3x3 = Conv1d_BN(x, nb_filter, 1, padding='same', strides=1, name=None)
            branch3x3 = Conv1d_BN(branch3x3, nb_filter, 3, padding='same', strides=1, name=None)

            branch5x5 = Conv1d_BN(x, nb_filter, 1, padding='same', strides=1, name=None)
            branch5x5 = Conv1d_BN(branch5x5, nb_filter, 1, padding='same', strides=1, name=None)

            branchpool = MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
            branchpool = Conv1d_BN(branchpool, nb_filter, 1, padding='same', strides=1, name=None)

            x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=2)
            return x  # 输出尺寸与输入一样 +pad


        cell_size = 512  # 隐藏层单元数


        def LSTM_d(x):
            x = LSTM(units=cell_size, kernel_regularizer=l2(0.005))(x)
            return x


        # GoogLeNet
        inpt = Input(shape=(a, 1))
        # padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = Conv1d_BN(inpt, 32, 7, strides=2, padding='same')
        # x = MaxPooling1D(pool_size=3, strides=2,padding='same')(x)

        x = Conv1d_BN(x, 64, 3, strides=1, padding='same')
        # x = MaxPooling1D(pool_size=3,strides=2,padding='same')(x)

        x = Inception(x, 8)  # 256
        # x = LSTM_d(x)
        # x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)

        x = Dropout(0.5)(x)
        # x = Inception(x,15)#480
        # x = MaxPooling1D(pool_size=3,strides=2,padding='same')(x)
        x = Inception(x, 16)  # 512
        # x = LSTM_d(x)
        x = Lambda(lambda x: B.expand_dims(x, axis=-1))(x)
        # x = Inception(x,16)
        # x = Inception(x,16)
        # x = Inception(x,17)#528
        # x = Inception(x,26)#832
        # x = MaxPooling1D(pool_size=3,strides=2,padding='same')(x)
        # x = Inception(x,16)
        # x = Inception(x,32)#1024
        # x = AveragePooling1D(pool_size=7,strides=7,padding='same')(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)  # 展开成一维
        x = Dense(256, activation='relu')(x)
        x = Dense(2, activation='softmax')(x)
        adam = Adam(lr=0.01)  # alpha：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）
        # sgd = SGD(lr=0.01)
        m = 23
        modelGoogleNet = Sequential()
        modelGoogleNet = Model(inpt, x, name='inception')
        modelGoogleNet.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 可改
        # loss='categorical_crossentropy' 'binary_crossentropy'
        # optimizer='sgd' 'adam' 'adagrad'
        modelGoogleNet.summary()  # 通过model.summary()输出模型各层的参数状况
#================================================CNN============================================================#
    if name == 'CNN':
        import numpy as np
        import pandas as pd
        from keras.utils import np_utils
        from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten, Reshape,BatchNormalization
        from sklearn import preprocessing
        from keras.models import Sequential
        from tensorflow.keras.optimizers import SGD, Adam
        from keras.regularizers import l2
        from keras.utils.vis_utils import plot_model
        import matplotlib.pyplot as plt
        from sklearn import metrics
        from sklearn.model_selection import KFold, train_test_split
        modelCNN = Sequential()
        # modelCNN.add(BatchNormalization())
        modelCNN.add(Conv1D(
            input_shape=(1, a),
            kernel_size=3,
            filters=4,  # 过滤器
            padding='same',  # 填充
            activation='relu',  # 激活函数
        ))

        modelCNN.add(MaxPool1D(
            pool_size=4,
            padding='same',
        ))
        # modelCNN.add(BatchNormalization())
        modelCNN.add(Conv1D(
            filters=16,  # 过滤器
            kernel_size=3,
            padding='same',  # 填充
            activation='relu',  # 激活函数
        ))
        modelCNN.add(MaxPool1D(
            pool_size=4,
            padding='same',
        ))
        # modelCNN.add(BatchNormalization())
        # modelCNN.add(Conv1D(
        #     filters=4,  # 过滤器
        #     kernel_size=2,
        #     padding='same',  # 填充
        #     activation='relu',  # 激活函数
        # ))
        # modelCNN.add(MaxPool1D(
        #     pool_size=4,
        #     padding='same',
        # ))
        # modelCNN.add(BatchNormalization())
        # modelCNN.add(Conv1D(
        #     filters=4,  # 过滤器
        #     kernel_size=3,
        #     padding='same',  # 填充
        #     activation='relu',  # 激活函数
        # ))
        # modelCNN.add(MaxPool1D(
        #     pool_size=2,
        #     padding='same',
        # ))

        modelCNN.add(Flatten())
        # modelCNN.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0001)))
        modelCNN.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0001)))
        modelCNN.add(Dense(32, activation='relu', kernel_regularizer=l2(0.0001)))
        modelCNN.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.0001)))  # 正则化项0.005
        adm = Adam(lr=0.0001)  # 学习率0.01
        modelCNN.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['acc'])
#===================================================ANN=================================================================#
    if name == 'ANN':
        import numpy as np
        import pandas as pd
        from keras.utils import np_utils
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from keras.optimizers import adam_v2
        from tensorflow.keras.optimizers import Adam
        from sklearn import preprocessing
        from keras.regularizers import l2
        import matplotlib.pyplot as plt
        from sklearn import metrics
        from sklearn.model_selection import KFold, train_test_split
        modelANN = Sequential([            Dense(units=512, input_dim=a, bias_initializer='one', activation='tanh', kernel_regularizer=l2(0.003)),
            # Dropout(0.4),#40%的神经元不工作
            Dense(units=128, bias_initializer='one', activation='tanh', kernel_regularizer=l2(0.003)),
            # Dropout(0.4),
            Dense(units=16, bias_initializer='one', activation='tanh', kernel_regularizer=l2(0.003)),
            # Dropout(0.4),
            Dense(units=2, bias_initializer='one', activation='softmax', kernel_regularizer=l2(0.003))
        ])

        adm = adam_v2.Adam(lr=0.001)
        modelANN.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['acc'])
# ================================================双lstm===================================================================#
    if name == '双LSTM':
        import time
        from keras.callbacks import ModelCheckpoint
        from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Reshape, Embedding, GRU, UpSampling1D, \
             ZeroPadding1D, Activation, Dropout, BatchNormalization, concatenate, advanced_activations, LeakyReLU
        from keras.layers import Dense
        from tensorflow.keras.optimizers import Adam
        from keras.utils import np_utils
        from sklearn.model_selection import train_test_split
        import numpy as np
        import matplotlib.pyplot as plt
        from keras.layers.wrappers import Bidirectional
        from keras.models import Sequential
        import tensorflow as tf
        import os
        import pandas as pd
        model_twoLSTM = Sequential()
        model_twoLSTM.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, return_sequences='True'),
                                input_shape=(3001, 1)))
        model_twoLSTM.add(LeakyReLU(alpha=0.05))
        model_twoLSTM.add(Dropout(0.5))

        model_twoLSTM.add(Bidirectional(LSTM(units=20, dropout=0.2, return_sequences='True')))
        model_twoLSTM.add(LeakyReLU(alpha=0.05))
        model_twoLSTM.add(Dropout(0.5))

        # model.add(Bidirectional(LSTM(units =20,return_sequences = 'True')))
        # model.add(LeakyReLU(alpha=0.05))
        model_twoLSTM.add(Flatten())
        model_twoLSTM.add(Dense(2, activation='softmax'))
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=3e-8)
        model_twoLSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# ================================================RNN===================================================================#
    if name == 'RNN':
        import numpy as  np
        import pandas as pd
        from keras.utils import np_utils
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras.layers.recurrent import LSTM
        from tensorflow.keras.optimizers import SGD, Adam
        from sklearn import preprocessing
        from keras.regularizers import l2
        # from keras.utils.vis_utils import plot_model
        from tensorflow.keras.utils import plot_model
        import matplotlib.pyplot as plt
        from sklearn import metrics
        from sklearn.model_selection import KFold, train_test_split
        cell_size = 512  # 隐藏层单元数
        input_size = a
        time_strep = 1
        modelRNN = Sequential()

        modelRNN.add(LSTM(
            units=cell_size,
            input_shape=(time_strep, input_size),
            kernel_regularizer=l2(0.001)
        ))
        modelRNN.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
        modelRNN.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.001)))
        adm = Adam(lr=0.005)
        modelRNN.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['acc'])






    if name == 'RNN':
        model = modelRNN
    if name == 'CNN':
        model = modelCNN
    if name == 'ANN':
        model = modelANN
    if name == 'AlexNet':
        model = modelAlexNet
    if name == 'ResNet':
        model = modelResNet
    if name == 'GoogleNet':
        model = modelGoogleNet
    if name == 'MCNN':
        model = modelMCNN
    if name == '双LSTM':
        model = model_twoLSTM

    return model




if __name__ == '__main__':
    # 设置交叉验证次数
    K = 5
    # a=241
    a = 96  # mfcc
    # a=30  #pls
    global count
    count = 0


    # 使用相对路径加载数据集
    # train_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\train_data.xlsx')
    # train_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Pls_train_data.xlsx')
    train_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\MFCC_train_data.xlsx')
    # test_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\test_data.xlsx')
    # test_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Pls_test_data.xlsx')
    test_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\MFCC_test_data.xlsx')


    # 划分数据集
    df_train = train_data.values[:, 1:]
    labels_train = train_data.values[:, 0]
    # labels_train = labels_train.T
    df_test = test_data.values[:, 1:]
    labels_test = test_data.values[:, 0]
    # labels_test = labels_test.T
    # df_train = df_train.reshape(1,-1,a)
    # df_test = df_test.reshape(1,-1,a)
    print(df_train.shape)
    print(df_test.shape)
    '''
        模型名称如下：
        'RNN'
        'CNN'
        'ANN'
        'AlexNet'
        'ResNet'
        'GoogleNet'
        'MCNN'
        '双LSTM'
       
    '''
    ModelName = 'CNN'
    model = ModelSelect(ModelName)
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='1D_CNN.png', show_shapes=True)
    cross_validation(df_train, labels_train, df_test, labels_test, K, model)
