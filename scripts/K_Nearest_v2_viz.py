"""
A simple implementation of Bayesian Classifier
@data: 2019.12.20
@author: Tingyu Mo
"""
import pandas as pd
import numpy as np
import os
import math
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,accuracy_score

class K_Nearest():
    '''
    A simple implementation of K Nearest algorithm
    '''
    def __init__(self,kN):
        self.kN = kN
        self.class_name = {}

    def data_generator(self,data_path):
        '''
        load data from .dat files and make preprocessing 
        '''
        df = pd.read_csv(data_path,header=None,encoding='utf-8',delimiter="\t",quoting=csv.QUOTE_NONE)
        df = df.iloc[3:,:] #去除头部说明信息
        data = df.values#将dataframe转成ndarray 内容为str
        data = data.tolist()#转成list
        features = []
        labels = []
        #数据集内容划分 分为features部分和labels部分
        print("preprocessing datasets!")
        for dataterm in tqdm(data):
            data_list = dataterm[0].split(',')
            data_list = [float(i)for i in data_list]
            features.append([i for i in data_list[:-1]])#除label以外所有的features
            labels.append(data_list[-1])
        # self.features,self.labels = np.array(features),np.array(labels)
        return np.array(features),np.array(labels)

    def dataset_split(self,train_data,train_target,test_size = 0.3,random_state=None):
        '''
        split datasets to training set and test set with predefined size
        '''
        # x_train,x_test, y_train, y_test = train_test_split(train_data,train_target,test_size=test_size, random_state=random_state)#划分数据集
        return train_test_split(train_data,train_target,test_size=test_size, random_state=random_state)

    def dataset_sparated(self,features,labels):
        '''
        sparate dataset into each class's data with labels
        return a dict inwhich the keys are class names and the values are ndarray[features,labels]
        '''
        print("sparating dataset!")
        sparated_dataset = dict()
        for label in tqdm(labels):
            label_str = str(label) #将标签转换为字典的key
            if label_str not in sparated_dataset:
                sparated_dataset[label_str] = None
            else: pass
        for key in sparated_dataset:        
            index = np.argwhere(labels == float(key))#利用标签获取该类的索引
            class_data = features[index]#利用该类的索引获取数据
            data_shape =class_data.shape
            if data_shape[1] == 1:
                class_data = class_data.reshape(data_shape[1],data_shape[0],data_shape[2])[0]
            class_label = labels[index]
            # sparated_dataset[key] = np.array([class_data,class_label])
            sparated_dataset[key] = (class_data,class_label)
        return sparated_dataset

    def cal_distance(self,samples,unknown_samples):
        '''
        caculate the euler distance between samples and unknown_samples
        '''
        difference = samples - unknown_samples # 利用numpy的proadcast机制求每一维的距离偏差
        distance = np.linalg.norm(difference,axis=1) #求每一行向量的二范数
        return distance
    
    def find_nearest(self,samples,distance,kN):
        '''
        search the nearest samples and return the position/index of those samples in datasets
        '''
        sorted_index = np.argsort(distance,kind='quicksort') #返回排序后的索引
        nearest_index = sorted_index[:kN] #取最近的kN个样本的索引
        return nearest_index


    def draw_kN(self,kN,samples,nearest_index):
        '''
        get the k nearest samples from datasets
        '''
        nearest_samples = samples[nearest_index]
        return nearest_samples

    def vote_for_classify(self,nearest_labels):
        '''
        using voting mechanism to caculate the most likely belonging class of unknown_samples
        '''
        mode_label = stats.mode(nearest_labels)[0][0] #计算出现最多的标签
        return mode_label

    def predict(self,x):
        '''
        make prediction for given unknown_samples
        '''
        print("prediction starts!")
        result = []
        for datapoint in tqdm(x):
            distance = self.cal_distance(self.x_train,datapoint)
            nearest_index = self.find_nearest(self.x_train,distance,self.kN)
            pred = self.vote_for_classify(self.y_train[nearest_index])
            result.append(pred)
        return np.array(result)

    def evaluate(self,y_pred,y_test):
        '''
        evaluate the accuracy and precision_score of K_Nearest classifiers
        '''
        print("evalutation starts!")
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        return acc,prec

    def viz(self, data_one,data_two,pre_data_one,pre_data_two):
        dot_size = 15
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("样本特征空间分布图")
        plt.scatter(data_one[:,0], data_one[:,1],marker='+',color='b',label='w1',s=dot_size)
        plt.scatter(data_two[:,0], data_two[:,1],marker='o',color='r',label='w2',s=dot_size)
        plt.scatter(pre_data_one[:,0], pre_data_one[:,1],marker='^',color='b',label='w1_pred',s=dot_size+20)
        plt.scatter(pre_data_two[:,0], pre_data_two[:,1],marker='v',color='r',label='w2_pred',s=dot_size+20)
        plt.legend(loc = 'upper right')
        plt.show()
        # self.plot_decision_boundary(X,y)
        # plt.show()

    def experiment_build(self,data_path,datasets = "banana"):
        if datasets == "banana":
            #加载数据集
            features,labels = self.data_generator(dataset_path)
            #划分训练集和测试集
            self.x_train,self.x_test, self.y_train, self.y_test = self.dataset_split(features,labels,test_size = 0.2)#split data
            sparated_train= self.dataset_sparated(self.x_train,self.y_train)
            w1_features,w1_labels = sparated_train["1.0"]
            w2_features,w2_labels = sparated_train["-1.0"]
            sparated_test = self.dataset_sparated(self.x_test,self.y_test)
            w1_features_pred,w1_labels = sparated_test["1.0"]
            w2_features_pred,w2_labels = sparated_test["-1.0"]
            #获取分类标签
            y_pred = self.predict(self.x_test)
            #计算准确率和精确度
            acc,prec = self.evaluate(y_pred,self.y_test)
            print("acc: {} prec:{}".format(acc,prec))
            self.viz(w1_features,w2_features,w1_features_pred,w2_features_pred)
        elif datasets == "normal":   
            w1_features = np.array([[0.2331, 2.3385], [1.5207, 2.1946], [0.6499, 1.6730], [0.7757, 1.6365],
                [1.0524, 1.7844], [1.1974, 2.0155], [0.2908, 2.0681], [0.2518, 2.1213],
                [0.6682, 2.4797], [0.5622, 1.5118], [0.9023, 1.9692], [0.1333, 1.8340],
                [-0.5431, 1.8704], [0.9407, 2.2948], [-0.2126, 1.7714], [0.0507, 2.3939],
                [-0.0810, 1.5648], [0.7315, 1.9329], [0.3345, 2.2027], [1.0650, 2.4568],
                [-0.0247, 1.7523], [0.1043, 1.6991], [0.3122, 2.4883], [0.6655, 1.7259],
                [0.5838, 2.0466], [1.1653, 2.0226], [1.2653, 2.3757], [0.8137, 1.7987],
                [-0.3399, 2.0828], [0.5152, 2.0798], [0.7226, 1.9449], [-0.2015, 2.3801],
                [0.4070, 2.2373], [-0.1717, 2.1614], [-1.0573, 1.9235], [-0.2099, 2.2604]])
            w2_features = np.array([[1.4010, 1.0298], [1.2301, 0.9611], [2.0814, 0.9154], [1.1655, 1.4901],
                        [1.3740, 0.8200], [1.1829, 0.9399], [1.7632, 1.1405], [1.9739, 1.0678],
                        [2.4152, 0.8050], [2.5890, 1.2889], [2.8472, 1.4601], [1.9539, 1.4334],
                        [1.2500, 0.7091], [1.2864, 1.2942], [1.2614, 1.3744], [2.0071, 0.9387],
                        [2.1831, 1.2266], [1.7909, 1.1833], [1.3322, 0.8798], [1.1466, 0.5592],
                        [1.7087, 0.5150], [1.5920, 0.9983], [2.9353, 0.9120], [1.4664, 0.7126],
                        [2.9313, 1.2833], [1.8349, 1.1029], [1.8340, 1.2680], [2.5096, 0.7140],
                        [2.7198, 1.2446], [2.3148, 1.3392], [2.0353, 1.1808], [2.6030, 0.5503],
                        [1.2327, 1.4708], [2.1465, 1.1435], [1.5673, 0.7679], [2.9414, 1.1288]])
            w1_labels = np.ones(w1_features.shape[0]) # 1 为w1类的label
            w2_labels = -1*np.ones(w2_features.shape[0])# -1 为w2类的label
            self.x_test = np.array([[1, 1.5], [1.2, 1.0], [2.0, 0.9], [1.2, 1.5], [0.23, 2.33]])
            self.x_train = np.vstack((w1_features,w2_features))
            self.y_train = np.hstack((w1_labels,w2_labels))
            #获取分类标签
            y_pred = self.predict(self.x_test)
            print("预测的标签结果为:",y_pred)
            sparated_train = self.dataset_sparated(self.x_train,self.y_train)
            w1_features,w1_labels = sparated_train["1.0"]
            w2_features,w2_labels = sparated_train["-1.0"]
            sparated_test = self.dataset_sparated(self.x_test,y_pred)
            w1_features_pred,w1_labels = sparated_test["1.0"]
            w2_features_pred,w2_labels = sparated_test["-1.0"]
            self.viz(w1_features,w2_features,w1_features_pred,w2_features_pred)
        
if __name__=="__main__":
    Root_dir = r'D:/Pattern_Recognion/Exp5-10'
    datasets_dir = os.path.join(Root_dir,"datasets")
    os.chdir(Root_dir)
    dataset_path = os.path.join(datasets_dir,'banana.dat')
    kN = 7 
    kN_Classifer = K_Nearest(kN)
    # kN_Classifer.experiment_build(dataset_path,datasets="banana")
    kN_Classifer.experiment_build(dataset_path,datasets="normal")
    


