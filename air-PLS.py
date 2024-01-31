import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
import data_obtain



def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The largxer lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
    i = np.arange(0, m)
    E = eye(m, format='csc')
    D = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * D.T * D))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=10, porder=1, itermax=100):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    '''
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): print('WARING max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z


def mean_spectrum_show(data1,data2):#计算平均光谱
    mean_health_spectrum=[]
    mean_iga_spectrum = []
    f=open(r'D:\mysoftware\py37\program\Arthritis\data\second-data（500-2500)\Arthritis_pingjunguagnpuzuobiao.xlsx')
    x=[]
    for line in f:
        x.append(line.strip('\n'))
    x=np.array(x)
    print(x)
    for i in range(len(data1[0])):
        #print(i)
        sum=0
        for j in range(len(data1)):
            sum=sum+data1[j][i]
        mean=sum/len(data1)
        mean_health_spectrum.append(mean)
    mean_health_spectrum = np.array(mean_health_spectrum)
    #print(mean_health_spectrum)
    for i in range(len(data2[0])):
        #print(i)
        sum=0
        for j in range(len(data2)):
            sum=sum+data2[j][i]
        mean=sum/len(data2)
        mean_iga_spectrum.append(mean)
    mean_iga_spectrum=np.array(mean_iga_spectrum)
    #print(mean_health_spectrum)
    '''fig, ax = pl.subplots(nrows=2, ncols=1)
    ax[0].plot(x,mean_health_spectrum)
    ax[0].set_title('mean health spectrum')
    ax[1].plot(x,mean_iga_spectrum)
    ax[1].set_title('mean iga spectrum')
    #pl.show()'''
    return x,mean_iga_spectrum,mean_health_spectrum

#mean_spectrum_show()


if __name__=='__main__':
    data1=[]
    data2=[]
    print ('Testing...')
    print ('Generating simulated experiment')
    x_train, y_train, x_test, y_test = data_obtain.gain_data("D:\mysoftware\py37\program\Arthritis\data\second-data（500-2500)\Arthritis_stadzong_data(500-1700).xlsx",
                                                             "D:\mysoftware\py37\program\Arthritis\data\second-data（500-2500)\Arthritis_stadzong_data(500-1700).xlsx")

    for i in x_train:
        data1.append(i-airPLS(i))
    data1=np.array(data1)

    for i in x_test:
          data2.append(i-airPLS(i))
    data2=np.array(data2)


    writer=pd.ExcelWriter('Arthritis_zong_data(500-2500)-airpls.xlsx')
    data1 = pd.DataFrame(data1)
    data1.insert(loc=0,column='-1',value=0)
    print(data1)

    data1.to_excel(writer,index=None,header=None)
    writer.save()
    writer.close()

    # writer = pd.ExcelWriter(r'Arthritis_illness_data(500-2500)-airpls.xlsx')
    # data2 = pd.DataFrame(data2)
    # data2.insert(loc=0, column='-1', value=1)
    # print(data2)
    #
    # data2.to_excel(writer, index=None, header=None)
    # writer.save()
    # writer.close()


