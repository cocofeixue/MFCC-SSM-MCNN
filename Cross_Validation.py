
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
    # plt.savefig(
    #     r'C:\Users\27152\Desktop\fanxiu\loss\\' + str(
    #     1) + '.jpg', dpi=300, bbox_inches="tight", pad_inches=0.0,transparent=True)

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
    #AleNet、ResNet、GoogleNet、twoLSTM
    # df_test = df_test.reshape(-1, a, 1)
    # CNN、RNN
    df_test = df_test.reshape(-1, 1, a)

    print("五折交叉验证的准确率如下：")
    print(np.array(df_train).shape,np.array(labels_train).shape)

    for i, (train, test) in enumerate(cv.split(df_train, labels_train)):#需要拆分y_trian

        global count
        count += 1
        cnt += 1
        print('第%d次迭代: ' % cnt)


        label_train = labels_train[train]
        y_train = np_utils.to_categorical(labels_train[train], num_classes=2)
        #AleNet、ResNet、GoogleNet、twoLSTM
        # data_train = df_train[train].reshape(-1,a,1)
        #CNN、RNN
        data_train = df_train[train].reshape(-1,1,a)
        #MCNN、ANN
        # data_train = df_train[train]


        label_val = labels_train[test]
        labels_val = np_utils.to_categorical(labels_train[test], num_classes=2)
        #AleNet、ResNet、GoogleNet、twoLSTM
        # df_val = df_train[test].reshape(-1,a,1)
        #CNN、RNN
        df_val = df_train[test].reshape(-1,1, a)
        # MCNN、ANN
        # df_val = df_train[test]

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

        test_predict1 = model.predict(data_train)
        test_predict1 = np.argmax(test_predict1, axis=1)
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
        # print(accuracy6)
        print('测试集F1-Score: ', accuracy6)
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
    plt.title('1D-CNN', font2)
    plt.legend(loc="lower right")
    plt.show()



    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 17,
             }


    mpl.rcParams["font.family"] = "SimHei"

    mpl.rcParams["font.size"] = "17"

    mpl.rcParams["axes.unicode_minus"] = False


    y_test = np.argmax(labels_test, axis=1)
    print("混淆矩阵值如下：")
    print(confusion_matrix(y_true=label_test, y_pred=predict_idx, labels=[0, 1]))
    # matrix就是二维ndarray数组类型。
    matrix = confusion_matrix(y_true=label_test, y_pred=predict_idx, labels=[0, 1])


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

    if name == 'CNN':
        model = modelCNN

    return model




if __name__ == '__main__':

    K = 5
    # a=241
    a = 96  # mfcc
    # a=30  #pls
    global count
    count = 0


    # train_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\train_data.xlsx')
    # train_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Pls_train_data.xlsx')
    train_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\MFCC_train_data.xlsx')
    # test_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\test_data.xlsx')
    # test_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\Pls_test_data.xlsx')
    test_data = pd.read_excel(r'D:\mysoftware\py37\program\Arthritis\data\second_data1700\Test_data\MFCC_test_data.xlsx')


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
        'CNN'

       
    '''
    ModelName = 'CNN'
    model = ModelSelect(ModelName)
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='1D_CNN.png', show_shapes=True)
    cross_validation(df_train, labels_train, df_test, labels_test, K, model)
