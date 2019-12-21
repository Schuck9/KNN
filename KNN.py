"""
A simple implementation of K-Nearest and K-Means algorithm
@data: 2019.12.21
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

class KNN():
    '''
    A simple implementation of K-Nearest algorithm
    '''
    def __init__(self,kN=7,centroids_num=5,method = "K_Nearest"):
        self.kN = kN
        self.centroids_num = centroids_num
        self.class_name = {}
        self.method = method

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

    def cal_means(self,x,axis=0):
        return x.mean(axis=axis)
    
    def cal_distance(self,samples,unknown_samples):
        '''
        caculate the euler distance between samples and unknown_samples
        '''
        difference = samples - unknown_samples # 利用numpy的proadcast机制求每一维的距离偏差
        distance = np.linalg.norm(difference,axis=1) #求每一行向量的二范数
        return distance
    
    def find_nearest(self,samples,distance,kN = 5):
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

    def generate_centroids(self,samples,samples_label= [],epochs = 100,viz = False):
        '''
        iterate samples to generate centroids until centroids maintain unchanged

        return centroids centroid's label and labels indicate each sample belonging to which centroid
        '''
        #获取随机的初始化聚点
        random_index = np.random.randint(0,samples.shape[0],self.centroids_num)
        centroids = samples[random_index]#variable
        centroids_index = np.array([range(centroids.shape[0])])[0]#constant
        centroids_label = np.array([range(centroids.shape[0])])[0]
        #生成样本的随机初始化聚类分配
        labels = np.random.randint(0,self.centroids_num,samples.shape[0])
        for epoch in tqdm(range(epochs)):
            #通过计算聚类与样本的距离更新各样本聚类归属
            for i ,datapoint in enumerate(samples):
                #计算每个聚点与待测样本距离
                distance = self.cal_distance(centroids,datapoint)
                #找到最近邻聚点
                nearest_index = self.find_nearest(self.x_train,distance,kN = 1)
                #更新待测样本的聚类归属
                labels[i] = centroids_index[nearest_index]
            #更新聚类位置
            for j ,centroid in enumerate(centroids):
                index = np.argwhere(labels == j)#利用标签获取该类的索引
                centroid_data = samples[index]#利用该类的索引获取数据
                data_shape = centroid_data.shape
                if data_shape[1] == 1:
                    centroid_data = centroid_data.reshape(data_shape[1],data_shape[0],data_shape[2])[0]
                centroids[j] = self.cal_means(centroid_data)
                if len(samples_label)!= 0:
                    #计算当前聚点到归属样本
                    distance = self.cal_distance(centroid_data,centroids[j])
                    #找到最近邻样本 假定用缺省的kN
                    nearest_index = self.find_nearest(centroid_data,distance)
                    centroids_label[j] = self.K_Nearest(centroids[j],centroid_data,samples_label,kN=5)
            if epoch%8 == 0:
                print("\n[{}|{}]".format(epoch+1,epochs))
                print("centroids:\n{}\ncentroids_label:\n{}".format(centroids,centroids_label))
                if viz == True:
                    self.viz_k_means(centroids,centroids_label,samples,labels)
        self.centroids = centroids
        self.centroids_label = centroids_label

        return centroids,centroids_label,labels
    

    def K_Nearest(self,unknown_sample,samples = [],samples_label =[],kN = None):
        '''
        using K_Nearest method to make prediction
        '''
        #设定kN的个数 即最近邻样本的个数
        if kN == None:
            kN = self.kN #没有临时更改kN则使用预先给定的kN
        if len(samples) == 0 or len(samples) == 0:
            samples = self.x_train
            samples_label = self.y_train
        
        #计算样本集的每个样本与待测样本距离
        distance = self.cal_distance(samples,unknown_sample)
        #找到最近邻样本
        nearest_index = self.find_nearest(samples,distance,self.kN)
        #判别待测样本的所属类
        pred = self.vote_for_classify(samples_label[nearest_index])
        
        return pred

    def K_Means(self,unknown_sample,centroids = None,kN = None):
        '''
        using K_Means method to make prediction
        '''
        #获取聚类结果 即均值点集
        if centroids == None:
            centroids = self.centroids #聚点集
            centroids_label = self.centroids_label
        #计算聚点集的聚点与待测样本距离
        distance = self.cal_distance(centroids,unknown_sample)
        #找到最近的一个聚类
        nearest_index = self.find_nearest(centroids,distance,1)
        #判别待测样本所属类
        pred = centroids_label[nearest_index][0]

        return pred

    def train(self,x_train=None,y_train=None):
        '''
        if method is k_means, calculate the centroids for after prediction
        '''
        if x_train == None or y_train == None:
            x_train ,y_train = self.x_train,self.y_train
        print("training starts!")
        if self.method == "K_Nearest":
            print("training finished!")
        elif self.method == "K_Means":
            centroids,centroids_label,labels = self.generate_centroids(x_train,y_train,viz = True)
            self.centroids,self.centroids_label,self.labels = centroids,centroids_label,labels
            return centroids,centroids_label,labels

    def predict(self,x):
        '''
        make prediction for given unknown_samples
        '''
        print("prediction starts!")
        result = []
        for datapoint in tqdm(x):
            if self.method == "K_Nearest":
                pred = self.K_Nearest(datapoint)
            elif self.method == "K_Means":
                pred = self.K_Means(datapoint)
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

    def viz(self, data_one,data_two,pre_data_one,pre_data_two,dot_size = 15):
        '''
        visualize the samples scatter
        '''
        if self.method == "K_Nearest":
            plt.rcParams['font.sans-serif']=['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.title("样本特征空间分布图")
            plt.scatter(data_one[:,0], data_one[:,1],marker='+',color='b',label='w1',s=dot_size)
            plt.scatter(data_two[:,0], data_two[:,1],marker='o',color='r',label='w2',s=dot_size)
            plt.scatter(pre_data_one[:,0], pre_data_one[:,1],marker='^',color='b',label='w1_pred',s=dot_size+20)
            plt.scatter(pre_data_two[:,0], pre_data_two[:,1],marker='v',color='r',label='w2_pred',s=dot_size+20)
            plt.legend(loc = 'upper right')
            plt.show()

    def viz_k_means(self,centroids,centroids_label,samples,labels,dot_size = 25):
        '''
        visualize centroids and samples belonging circumstances
        '''
        if self.method == "K_Means":
            color_map = ["b","r","g","c","k","m","y"]
            marker_map =["+","o","^","v","8","s","p","h","p"]
            plt.rcParams['font.sans-serif']=['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.title("聚点及样本点空间分布图 聚点数量:{}".format(self.centroids_num))

            plt.scatter(centroids[:,0],centroids[:,1],c = centroids_label,marker=marker_map[-1],label="centroids",s=dot_size+45)
            
            sparated_dataset = self.dataset_sparated(samples,labels)
            for i ,key in enumerate(sparated_dataset.keys()):
                features,labels = sparated_dataset[key]
                plt.scatter(features[:,0],features[:,1],marker=marker_map[i],color=color_map[i],label=key,s=dot_size)
            plt.legend(loc = 'upper right')
            plt.show()

    def experiment_build(self,data_path,datasets = "banana"):
        '''
        build experiment to illustrate algorithm's performance
        '''
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
            #模型训练
            self.train()
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
            #模型训练
            self.train()
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
    centroids_num=7
    kN_Classifer = KNN(centroids_num,method="K_Means")
    # kN_Classifer = KNN(kN,method="K_Nearest")
    kN_Classifer.experiment_build(dataset_path,datasets="banana")
    # kN_Classifer.experiment_build(dataset_path,datasets="normal")
    


