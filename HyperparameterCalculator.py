import math
from validate import validate
import matplotlib.pyplot as plt
from Train import *


def Hyperprameter_train_model(train_X,train_Y,c) :
    svc=SVC(C=c,kernel='rbf',gamma='scale') 
    svc.fit(train_X,train_Y)
    return svc


def hypermaterKnowing_C_value(train_X,train_Y,validation_split_percent):
    length_training_Data_X=math.floor(((float(100-validation_split_percent))/100)*len(train_X))
    training_Data_X=train_X[0:length_training_Data_X]
    training_Data_Y=train_Y[0:length_training_Data_X]
    testing_Data_X = train_X[length_training_Data_X:]
    Actual_Data_Y = train_Y[length_training_Data_X:]
    max_depth=[i for i in range(1,len(train_X[0])+4)]
    min_size=[i for i in range(len(list(set(train_Y)))+2)]
    for j in range(1,20):
        model=Hyperprameter_train_model(training_Data_X,training_Data_Y,j/2)
        pred_Y = model.predict(testing_Data_X)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(Actual_Data_Y, pred_Y)
        print(" this is accuracy for above C value---",j/2)
        print("Accuracy", accuracy)

X,Y=Import_data()
X=data_processing(X,Y)
hypermaterKnowing_C_value(X,Y,30)