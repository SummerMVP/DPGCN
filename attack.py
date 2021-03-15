from classifier import train as train_model,shadow_train,trainclf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
# import theano
import argparse
import os
import imp
import tensorflow as tf
from math import isnan
import numpy as np
import math
np.random.seed(21312)
MODEL_PATH = './model/'
DATA_PATH = './data/'

import DPGCN.train as target_train

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


def load_attack_data():
    fname = MODEL_PATH + 'attack_train_data.npz'
    with np.load(fname) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.npz'
    with np.load(fname) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32')


def train_target_model(dataset,save,DP,epoch):
    attack_x, attack_y, classes=target_train.train(dataset,DP,epochs=epoch,model=args.target,target_eps=0.194,clip=1e-4)
    # attack_x, attack_y, classes,_=shadow_train(dataset,epoch,0,hidden1=16,model=args.target)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes=classes.astype('int32')
    if save:
        np.savez(MODEL_PATH + 'attack_test_data.npz', attack_x, attack_y)
    return attack_x, attack_y, classes

# 这里的阴影模型是和原始目标模型一致的
def train_shadow_models(n_shadow,dataset,epoch,save):
    # for attack model
    attack_x, attack_y = [], []
    attack_i_x, attack_i_y = [], []
    classes = []
    classes_i=[]
    acc=[]
    hidden1=20
    S=0.1
    for i in range(n_shadow):  
        attack_i_x,attack_i_y,classes_i,acc_i=shadow_train(dataset,epoch,hidden1=hidden1,model=args.target,num=i)
        # attack_x, attack_y, classes=target_train.train(args.DP,epochs=epoch,model=args.target)
        attack_x.append(attack_i_x)
        attack_y.append(attack_i_y)
        classes.append(classes_i)
        # acc.append(acc_i)
        # if(i>1 and acc[-1]-acc[-2]<S):
        #     hidden1-=1
        # elif(i>1 and acc[-2]-acc[-1]<S):
        #     hidden1+=1

    attack_x = np.vstack(attack_x)
    attack_y = np.vstack(attack_y)
    classes=np.vstack(classes)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes=classes.astype('int32')
    # print("attack_y:",attack_x.shape)
    if save:
        np.savez(MODEL_PATH + 'attack_train_data.npz', attack_x, attack_y)
    return attack_x, attack_y, classes


def train_attack_model(classes, dataset=None, n_hidden=20, learning_rate=0.01, batch_size=200, epochs=50,
                       model='nn', l2_ratio=1e-7):
    if dataset is None:
        dataset = load_attack_data()
    train_x, train_y, test_x, test_y = dataset
    train_classes, test_classes = classes

    # 打乱索引,这里打乱数据 
    np.random.seed(12)
    index1 = [i for i in range(len(train_x))] 
    np.random.shuffle(index1) 
    train_x = train_x[index1] 
    train_y = train_y[index1] 
    train_classes=train_classes[index1]

    np.random.seed(913)     
    index2 = [i for i in range(len(test_x))]      
    np.random.shuffle(index2)      
    test_x = test_x[index2]      
    test_y = test_y[index2]
    test_classes=test_classes[index2] 

    # print("train_x len:",len(train_x))
    # print("train_x:",train_x.shape)
    # print("class len:",len(train_classes))
    # print("train_classes:",train_classes.shape)
    # 分片，起点为0，终点是len
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes =train_x.shape[1]
    # print("unique_classes",train_x.shape[1])
    # true_y = []
    # pred_y = []
    # a = np.argmax(train_x, axis=1) 
    # b = np.argmax(test_x, axis=1) 
    # print("a：",a)

    train_x=np.hstack((train_x,train_classes))
    test_x=np.hstack((test_x,test_classes))

    print("train_x:",train_x.shape)
    print("test_x:",test_x.shape)
    acc=[]
    acc_map={}
    nodata=0
    for c in range(unique_classes):
        print ('Training attack model for class {}...'.format(c))
        c_train_indices=[]
        c_test_indices=[]
        for i in range(len(train_classes)):
            if(train_classes[i]==c):
                c_train_indices.append(i)
        for i in range(len(test_classes)):
            if(test_classes[i]==c):
                c_test_indices.append(i)
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
        if(len(c_train_x)!=0):
            c_acc_y = train_model(c_dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate)
            # c_acc_y=trainclf(c_dataset)
            if(isnan(c_acc_y)):
                print(str(c)+" class accuracy is wrong!")
                print("len(test_x test_y):",len(c_test_x),len(c_test_y))
            else:
                acc.append(c_acc_y*len(c_train_x)/len(train_x))
                acc_map[str(c)]=c_acc_y
            print("the "+str(c)+" model is OK!")
        else:
            nodata+=1
            print(str(c)+" don't have any train data!")
    print ('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
    print("test accuracy:",acc_map)
    # print ('Testing Average Accuracy: {}'.format(np.mean(acc)))
    print ('Testing sum Accuracy: {}'.format(sum(acc)/np.sqrt(1+math.log(nodata+1,math.e))))
    


def load_data(data_name):
    with np.load(DATA_PATH + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y, test_x, test_y


def attack_experiment():
    print ('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')
    # dataset = load_data('target_data.npz')
    attack_test_x, attack_test_y, test_classes = train_target_model(args.dataset,args.target_save,args.DP,args.target_epochs)
    print ('-' * 10 + 'TRAIN SHADOW' + '-' * 10 + '\n')
    attack_train_x, attack_train_y, train_classes = train_shadow_models(args.n_shadow,args.dataset,args.shadow_epochs,args.shadow_save)
    print ('-' * 10 + 'TRAIN ATTACK' + '-' * 10 + '\n')
    dataset = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    train_attack_model(
        dataset=dataset,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        learning_rate=args.attack_learning_rate,
        n_hidden=args.attack_n_hidden,
        l2_ratio=args.attack_l2_ratio,
        model=args.attack_model,
        classes=(train_classes, test_classes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # target model
    # shadow model
    parser.add_argument('--target', type=str, default='gcn') 
    parser.add_argument('--n_shadow', type=int, default=10)
    parser.add_argument('--dataset', type=str, default="pubmed")
    parser.add_argument('--DP', type=int, default=1)
    parser.add_argument('--shadow_epochs', type=int, default=40)
    parser.add_argument('--target_epochs', type=int, default=500)
    parser.add_argument('--target_save', type=int, default=1)
    parser.add_argument('--shadow_save', type=int, default=1)
   
    # attack model configuration
    parser.add_argument('--attack_model', type=str, default='softmax')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=20)
    parser.add_argument('--attack_epochs', type=int, default=100)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)

    # parse configuration
    args = parser.parse_args()
    if(args.dataset=='cora'):
        args.target_epochs=120
    elif(args.dataset=='citeseer'):
        args.target_epochs=200
    elif(args.dataset=='pubmed'):
        args.target_epochs=200

    attack_experiment()

