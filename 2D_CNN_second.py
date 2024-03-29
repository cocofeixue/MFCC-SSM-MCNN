# -*- coding:utf-8 -*-
"""
Author:Xu
Time:2022/03/24
"""

import numpy as np 
import pandas as pd 
import os 
import cv2 
import matplotlib.pyplot as plt 
import random as rdm 
from sklearn.preprocessing import LabelEncoder 
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split 
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
from keras.preprocessing.image import ImageDataGenerator
# warnings.filterwarnings(action='ignore')
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf
k = 5
# a=(150,150,3)
print(cv2.__file__)

# # GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "/gpu:0"   

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# session = tf.compat.v1.Session(config=config)



# rotation_range 
# zoom_range 
# width_shift_range 
# height_shift_range
# horizontal_filp 
# vertical_filp
# data_augment = ImageDataGenerator(rotation_range= 10,zoom_range= 0.1,
#                                   width_shift_range = 0.1,height_shift_range = 0.1,
#                                   horizontal_flip = False, vertical_flip = False)

def show_history(history):
    loss = history.history['loss'] 
    val_loss = history.history['val_loss'] 
    epochs = range(1, len(loss) + 1) 
    plt.figure(figsize=(12,4)) 
    plt.subplot(1, 2, 1) 
    plt.plot(epochs, loss, 'r', label='Training loss') 
    plt.plot(epochs, val_loss, 'b', label='Validation loss') 
    plt.title('Training and validation loss') 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.legend() 
    acc = history.history['acc'] 
    val_acc = history.history['val_acc'] 
    plt.subplot(1, 2, 2) 
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc') 
    plt.title('Training and validation accuracy') 
    plt.xlabel('Epochs') 
    plt.ylabel('Accuracy') 
    plt.legend()
    plt.show() 

def cross_validation(df_train, labels_train, df_test, labels_test, k, model):
    

    test_predict = []


    test_precision = []


    test_recall = []



    test_F1_Score = []


    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

  
    cnt = 0

    allacc1 = 0
    allacc2 = 0
    allacc3 = 0
 
    label_test = labels_test
    labels_test = np_utils.to_categorical(labels_test, num_classes=2)
    df_test=df_test.reshape(-1,100,100,3)
    #AleNet、ResNet、GoogleNet、twoLSTM
    # df_test = df_test.reshape(-1, a, 1)
    # CNN、RNN
    # df_test = df_test.reshape(-1, 1, a)
    # print("十折交叉验证的准确率如下：")
    # print(df_train)
    # print(labels_train)
    # print("十折交叉验证的准确率如下：")
    # print(np.array(df_train).shape)
    # print(np.array(labels_train).shape)
    # print(df_train)
    # print(labels_train)

    count =0
    for i, (train, test) in enumerate(cv.split(df_train, labels_train)):#需要拆分y_trian


        count += 1
        cnt += 1
        print('第%d次迭代: ' % cnt)


        label_train = np.array(labels_train)[train]
        # y_train = label_encoder.fit_transform(label_train)
        # y_train = to_categorical(labels_train[train],4)
        y_train = np_utils.to_categorical(np.array(labels_train)[train], num_classes=2)
        data_train =df_train[train]
        print(data_train.shape)

        data_train =df_train[train].reshape(-1,100,100,3)
        #AleNet、ResNet、GoogleNet、twoLSTM
        # data_train = df_train[train].reshape(-1,a,1)
        #CNN、RNN
        # data_train = df_train[train].reshape(-1, 1, a)
        #MCNN、ANN
        # data_train = df_train[train]

        # y_train = label_encoder.fit_transform(y_train)
        # y_train = to_categorical(y_train,4)
        # y_test = label_encoder.fit_transform(y_test)
        # y_test = to_categorical(y_test,4)


        label_val = np.array(labels_train)[test]
        # labels_val = label_encoder.fit_transform(label_val)
        # labels_val = to_categorical(labels_train[test],4)
        labels_val = np_utils.to_categorical(np.array(labels_train)[test], num_classes=2)
        df_val = df_train[test]
        df_val = df_train[test].reshape(-1,100,100,3)
        #AleNet、ResNet、GoogleNet、twoLSTM
        # df_val = df_train[test].reshape(-1,a,1)
        #CNN、RNN
        # df_val = df_train[test].reshape(-1, 1, a)
        # MCNN、ANN
        # df_val = df_train[test]

        # print(np.array(df_val).shape)
        # print(np.array(labels_val).shape)
        # filepath = r'./best_model' + str(count) + '.h5'
        # checkpoint = ModelCheckpoint(filepath,  # (就是你准备存放最好模型的地方),
        #                              monitor='val_acc',  # (或者换成你想监视的值,比如acc,loss, val_loss,其他值应该也可以,还没有试),
        #                              verbose=1,  # (如果你喜欢进度条,那就选1,如果喜欢清爽的就选0,verbose=冗余的),
                                     # save_best_only='True',  # (只保存最好的模型,也可以都保存),
                                     # save_weights_only='True',  # 只保存权重
                                     # mode='max',
                                     # (如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
                                     # period=1)  # (checkpoints之间间隔的epoch数)
        # history = model.fit(data_augment.flow(data_train, y_train, batch_size=23), batch_size=23, epochs=100, verbose=2, shuffle=True,
        #                     validation_data=(df_val, labels_val), callbacks=[checkpoint])
        history = model.fit(data_train, y_train, batch_size=30, epochs=100, verbose=2,
                            shuffle=True,validation_data=(df_val, labels_val),)# callbacks=[checkpoint]
        # model.load_weights(r'./best_model' + str(count) + '.h5')

        show_history(history)

        # model.fit(data_train, y_train, batch_size=32, validation_data=(df_val, labels_val), epochs=10)  # 批处理个数32
        # print(df_val.shape)
        # print(labels_val)
        # print(data_train)
        # print(y_train)
  
        test_predict1 = model.predict(data_train)
        test_predict1 = np.argmax(test_predict1, axis=1)
        # print(test_predict1)
        # print(test_predict1)
        accuracy1 = accuracy_score(label_train, test_predict1)
        print('训练集准确率: ', accuracy1)
        allacc1 = allacc1 + accuracy1

 
        test_predict2 = model.predict(df_val)
        test_predict2 = np.argmax(test_predict2, axis=1)
        accuracy2 = accuracy_score(label_val, test_predict2)
        print('验证集准确率: ', accuracy2)
        allacc2 = allacc2 + accuracy2

  
        test_predict3 = model.predict(df_test)
        test_predict3 = np.argmax(test_predict3, axis=1)
        accuracy3 = accuracy_score(label_test, test_predict3)
        print('测试集准确率: ', accuracy3)
        allacc3 = allacc3 + accuracy3



        predict_p = model.predict(df_test)
        test_predict.append(predict_p)

 
        accuracy4 = precision_score(label_test, test_predict3, average=None)
        test_precision.append(accuracy4)

 
        accuracy5 = recall_score(label_test, test_predict3, average=None)
        test_recall.append(accuracy5)


        accuracy6 = f1_score(label_test, test_predict3, average=None)
        test_F1_Score.append(accuracy6)


    mean_predict = test_predict[0]
    for i in range(1, 5):
        mean_predict += test_predict[i]
    mean_predict = mean_predict / 5


    predict_idx = np.argmax(mean_predict, axis=1)


    mean_precision = test_precision[0]
    for i in range(1, 5):
        mean_precision += test_precision[i]
    mean_precision = mean_precision / 5


    mean_recall = test_recall[0]
    for i in range(1, 5):
        mean_recall += test_recall[i]
    mean_recall = mean_recall / 5


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




    n_classes = 2

    y_score = mean_predict


    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 17,
             }
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 将数据转化为矩阵
    y_test = np.array(labels_test)
    # print(y_test)
    y_test = label_binarize(y_test, classes=[0, 1])
    y_test = np_utils.to_categorical(y_test, num_classes=2)  # 将标签转化为形如(样本数, 类别数)的二值序列。
    # print(y_test)
    for i in range(n_classes):
        # [:, i]取i列的所有行，
        fpr[i], tpr[i], _ = roc_curve(labels_test[:, i], y_score[:, i])
        # ROC曲线下的面积大小
        roc_auc[i] = auc(fpr[i], tpr[i])


    fpr["micro"], tpr["micro"], _ = roc_curve(labels_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # #MFCC-CNN
    # Note = open('MFCC-CNN_fpr.txt', mode='w')
    # for i in fpr["micro"]:
    #    Note.write(str(i)+'\n')
    # Note.close()
    # Note = open('MFCC-CNN_tpr.txt', mode='w')
    # for i in tpr["micro"]:
    #     Note.write(str(i) + '\n')
    # Note.close()
    # #
    # # STFT-CNN
    # Note = open('STFT-CNN_fpr.txt', mode='w')
    # for i in fpr["micro"]:
    #    Note.write(str(i)+'\n')
    # Note.close()
    # Note = open('STFT-CNN_tpr.txt', mode='w')
    # for i in tpr["micro"]:
    #     Note.write(str(i) + '\n')
    # Note.close()

    # #MFCC-MCNN
    # Note = open('MFCC-OFN8_fpr.txt', mode='w')
    # for i in fpr["micro"]:
    #    Note.write(str(i)+'\n')
    # Note.close()
    # Note = open('MFCC-OFN8_tpr.txt', mode='w')
    # for i in tpr["micro"]:
    #     Note.write(str(i) + '\n')
    # Note.close()



    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

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
    plt.title('STFT-CNN', font2)
    plt.legend(loc="lower right")
    plt.show()



    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 17,
             }


    mpl.rcParams["font.family"] = "SimHei"

    mpl.rcParams["font.size"] = "17"

    mpl.rcParams["axes.unicode_minus"] = False

。
    y_test = np.argmax(labels_test, axis=1)
    print("混淆矩阵值如下：")
    print(confusion_matrix(y_true=label_test, y_pred=predict_idx, labels=[0, 1]))
    # matrix就是二维ndarray数组类型。
    matrix = confusion_matrix(y_true=label_test, y_pred=predict_idx, labels=[0, 1])


    plt.matshow(matrix, cmap=plt.cm.Blues, alpha=0.5)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            plt.text(x=j, y=i, s=matrix[i, j], va='center', ha='center')
    plt.xlabel('Predicted Condition', font2)
    plt.ylabel('True Condition', font2)
    plt.title('STFT-CNN', font2)
    plt.show()


print(os.listdir(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Up_data')) #打印目录结构
# stad_dir=r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\2data'  #根目录

#stft
# train_health_dir=r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Up_data\STFT_train_0_data' #train健康目录
# train_illness_dir=r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Up_data\STFT_train_1_data' #train患病目录
# test_health_dir=r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Up_data\STFT_test_0_data'  #test健康目录
# test_illness_dir=r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Up_data\STFT_test_1_data' #test患病目录
#MFCCs
train_health_dir=r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Up_data\MFCCs_train_0_data' #train健康目录
train_illness_dir=r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Up_data\MFCCs_train_1_data' #train患病目录
test_health_dir=r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Up_data\MFCCs_test_0_data'  #test健康目录
test_illness_dir=r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Up_data\MFCCs_test_1_data' #test患病目录

# X = [] #初始化
# y_label = [] #初始化
imgsize = 100 #图片大小


# def training_data(label,data_dir):
#     print ("正在读入：", data_dir)
#     for img in os.listdir(data_dir):
#         path = os.path.join(data_dir,img) 
#         img = cv2.imread(path,cv2.IMREAD_COLOR) 
#         img = cv2.resize(img,(imgsize,imgsize)) 
#         X.append(np.array(img)) 
#         y_label.append(str(label)) 

# training_data('0',health_dir)
# training_data('1',illness_dir) 
x_train = []
y_train = []
x_test = []
y_test = []
print(os.listdir(train_illness_dir))
def create_data(label1,train_data_dir1,label2,train_data_dir2,test_data_dir1,test_data_dir2):
    for img1 in os.listdir(train_data_dir1):
        path1 = os.path.join(train_data_dir1,img1)
        img1 = cv2.imread(path1, cv2.IMREAD_COLOR) 
        img1 = cv2.resize(img1, (imgsize, imgsize))  
        img1 = tf.cast(img1, tf.float32)
        x_train.append(np.array(img1))
        y_train.append(int(label1))
    for img2 in os.listdir(train_data_dir2):
        path2 = os.path.join(train_data_dir2,img2)
        img2 = cv2.imread(path2,cv2.IMREAD_COLOR)
        img2 = cv2.resize(img2, (imgsize, imgsize)) 
        img2 = tf.cast(img2, tf.float32)
        x_train.append(np.array(img2))
        y_train.append(int(label2))
    for img1 in os.listdir(test_data_dir1):
        path1 = os.path.join(test_data_dir1,img1)
        img1 = cv2.imread(path1, cv2.IMREAD_COLOR)  
        img1 = cv2.resize(img1, (imgsize, imgsize))  
        img1 = tf.cast(img1, tf.float32)
        x_test.append(np.array(img1))
        y_test.append(int(label1))
    for img2 in os.listdir(test_data_dir2):
        path2 = os.path.join(test_data_dir2, img2)
        img2 = cv2.imread(path2, cv2.IMREAD_COLOR)
        img2 = cv2.resize(img2, (imgsize, imgsize)) 
        img2 = tf.cast(img2, tf.float32)
        x_test.append(np.array(img2))
        y_test.append(int(label2))
    return x_train,y_train,x_test,y_test
create_data(0,train_health_dir,1,train_illness_dir,test_health_dir,test_illness_dir)


# fig,ax=plt.subplots(5,2) #画布
# fig.set_size_inches(15,15) #大小
# for i in range(5):
#     for j in range (2):
#         r=rdm.randint(0,len(X)) #随机选择图片
#         ax[i,j].imshow(X[r]) #显示图片
#         ax[i,j].set_title('Spectrograms: '+y_label[r]) #IGA的类别
# plt.tight_layout() #绘图


# def training_data(label,data_dir):
#     print ("正在读入：", data_dir)
#     for img in os.listdir(data_dir): #目录
#         path = os.path.join(data_dir,img) #目录+文件名
#         img = cv2.imread(path,cv2.IMREAD_COLOR) #读入图片
#         img = cv2.resize(img,(imgsize,imgsize)) #设定图片像素维度
#         X.append(np.array(img)) #X特征集
#         y_label.append(str(label)) #y标签，即类别

# from tensorflow.keras import layers # 导入所有层 行1
# from tensorflow.keras import models # 导入所有模型 行2
# cnn = models.Sequential() # 贯序模型 行3
# #2D-CNN
# cnn.add(layers.Conv2D(4, (3, 3), activation='relu', # 输入卷积层
#                         input_shape=(100,100,3)))
# cnn.add(layers.Conv2D(8, (3, 3), activation='relu',name='Conv1', # 输入卷积层
#                         strides= 2))
# cnn.add(layers.MaxPooling2D((4,4))) # 最大池化层
# cnn.add(layers.Conv2D(16, (3, 3), activation='relu',name='Conv2')) # 卷积层
# # cnn.add(layers.MaxPooling2D((2, 2))) # 最大池化层
# # cnn.add(layers.Conv2D(16, (3, 3), activation='relu',strides= 2)) # 卷积层
# cnn.add(layers.MaxPooling2D((4,4))) # 最大池化层
# cnn.add(layers.Flatten()) # 展平层
# cnn.add(layers.Dense(128, activation='relu')) # 全连接层
# # cnn.add(layers.Dense(64, activation='relu')) # 全连接层
# cnn.add(layers.Dense(32, activation='relu')) # 全连接层
# cnn.add(layers.Dense(2, activation='softmax')) # 分类输出层
# cnn.compile(loss='categorical_crossentropy', # 损失函数
#             # optimizer='RMSprop', # 优化器
#             optimizer='adam',
#             metrics=['acc']) # 评估指标
# model = cnn
# cross_validation(x_train,y_train,x_test,y_test,k,model)

#2D-MCNN
from keras import Sequential, Model
import keras.layers as KL
import keras.backend as K
import numpy as np
from keras.utils.vis_utils import plot_model
from tensorflow.keras import optimizers
from keras import losses
from keras import metrics

input_tensor=KL.Input((100,100,3))
x1=KL.Conv2D(8,(3,3),padding="same")(input_tensor)
x1=KL.BatchNormalization()(x1)
x1=KL.Activation("relu")(x1)
x1=KL.Conv2D(8,(3,3),padding="same")(x1)
x1=KL.Activation("relu")(x1)
x1=KL.MaxPooling2D(4,(3,3))(x1)
F1=KL.Flatten()(x1)

x2=KL.Conv2D(16,(3,3),padding="same")(x1)
x2=KL.BatchNormalization()(x2)
x2=KL.Activation("relu")(x2)
x2=KL.MaxPooling2D(4,(3,3))(x2)
F2=KL.Flatten()(x2)

x3=KL.Conv2D(16,(3,3),padding="same")(x2)
x3=KL.BatchNormalization()(x3)
x3=KL.Activation("relu")(x3)
x3=KL.MaxPooling2D(4,(3,3))(x3)
F3=KL.Flatten()(x3)

xx=KL.concatenate([F1,F2,F3])
y=KL.Dense(64, activation='relu')(xx)
y=KL.Dense(32, activation='relu')(y)
y=KL.Dense(2,activation='softmax')(y)
model = Model(inputs=[input_tensor],outputs=[y])
model.compile(loss='categorical_crossentropy', # 损失函数 行15
            # optimizer='RMSprop', # 优化器
            optimizer='adam',
            metrics=['acc']) # 评估指标
cross_validation(x_train,y_train,x_test,y_test,k,model)

# for i in range(len(model.layers)):
# 	layer = model.layers[i]
# 	# check for convolutional layer
# 	if 'conv' not in layer.name:
# 		continue
# 	# summarize output shape
# 	print(i, layer.name, layer.output.shape)
#
#
# def get_row_col(num_pic):
#     squr = num_pic ** 0.5
#     row = round(squr)
#     col = row + 1 if squr - row > 0 else row
#     return row, col
# def visualize_feature_map(img_batch):
#     feature_map = np.squeeze(img_batch, axis=0)
#     print(feature_map.shape)
#
#     feature_map_combination = []
#     plt.figure()
#
#     num_pic = feature_map.shape[2]
#     row, col = get_row_col(num_pic)
#
#     for i in range(0, num_pic):
#         feature_map_split = feature_map[:, :, i]
#         feature_map_combination.append(feature_map_split)
#         plt.subplot(row, col, i + 1)
#         plt.imshow(feature_map_split)
#         plt.axis('off')
#         plt.title('feature_map_{}'.format(i))
#
#     # plt.savefig('feature_map.png')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    #
    # # 各个特征图按1：1
    # feature_map_sum = sum(ele for ele in feature_map_combination)
    # plt.imshow(feature_map_sum)
    # # plt.savefig("feature_map_sum.png")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.colorbar()
    # plt.show()
# #
# from keras.preprocessing.image import img_to_array
# from keras.models import Model
# from matplotlib import pyplot
#
# model1 = Model(inputs=model.input, outputs=model.get_layer('Conv1').output)
# # model1 = Model(inputs=model.input, outputs=model.get_layer('Conv2').output)
#
#
# # average_Mel_path = r'D:\mysoftware\py37\program\Arthritis\data\stad_2D_figure\Stad_data\average_Mel'
# average_MFCCs_path = r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\pingjun'
# # average_STFT_path = r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\pingjun-STFT'
# # for img1 in os.listdir(average_Mel_path):
# for img1 in os.listdir(average_MFCCs_path):
# # for img1 in os.listdir(average_STFT_path):
#     path1 = os.path.join(average_MFCCs_path, img1)
#     img1 = cv2.imread(path1, cv2.IMREAD_COLOR)  # 读入图片
#     img = cv2.resize(img1, (100, 100))  # 设定图片像素维度
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     feature_maps = model1.predict(img)
#     visualize_feature_map(feature_maps)










