import csv
import pickle
import numpy as np
from sklearn.svm import SVC

mean_null_list=[]
All_mini_maxi_mean_normalize_value=[]
threshold1=0.0
threshold2=0.5

def Import_data():
    X=np.genfromtxt("train_X_svm.csv",delimiter=',',dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_svm.csv",delimiter=',',dtype=int)
    return X,Y

def replace_null_values_with_mean(X):
    mean_of_nan=np.nanmean(X,axis=0)
    mean_null_list.append(mean_of_nan)
    index=np.where(np.isnan(X))
    X[index]=np.take(mean_of_nan,index[1])
    return X

def mean_normalize(X, column_indices):
    mini=np.min(X[:,column_indices],axis=0)
    maxi=np.max(X[:,column_indices],axis=0)
    mean=np.mean(X[:,column_indices],axis=0)
    All_mini_maxi_mean_normalize_value.append([mini,maxi,mean])
    X[:,column_indices]=(X[:,column_indices]-mean)/(maxi-mini)
    return X

def get_correlation_matrix(X,class_Y):
    given_X=np.hstack((class_Y.reshape(len(X),1),X))
    num_vars = len(given_X[0])
    m = len(X)
    correlation_matix = np.zeros((num_vars,num_vars))
    for i in range(0,num_vars):
        for j in range(i,num_vars):
            mean_i = np.mean(given_X[:,i])
            mean_j = np.mean(given_X[:,j])
            std_dev_i = np.std(given_X[:,i])
            std_dev_j = np.std(given_X[:,j])
            numerator = np.sum((given_X[:,i]-mean_i)*(given_X[:,j]-mean_j))
            denominator = (m)*(std_dev_i)*(std_dev_j)
            corr_i_j = numerator/denominator    
            correlation_matix[i][j] = corr_i_j
            correlation_matix[j][i] = corr_i_j
    return correlation_matix


def select_features(corr_mat, T1, T2):
    filter_feature=[]
    m=len(corr_mat)
    for i in range(1,m):
        if(abs(corr_mat[i][0])>T1):
            filter_feature.append(i-1)
    removed_feature=[]
    n=len(filter_feature)
    select_features=list(filter_feature)
    for i in range(0,n):
        for j in range(i+1,n):
            f1=filter_feature[i]
            f2=filter_feature[j]
            if (f1 not in removed_feature) and (f2 not in removed_feature):
                if(abs(corr_mat[f1][f2])>T2):
                    select_features.remove(f2)
                    removed_feature.append(f2)
                    
    return select_features

def data_processing(class_X,class_Y) :
    X=replace_null_values_with_mean(class_X)
    for i in range(class_X.shape[1]):
        X=mean_normalize(X,i)
    
    correlation_matrix= get_correlation_matrix(X,class_Y)
    selected_feature_list=select_features(correlation_matrix,threshold1,threshold2)
    X=X[:,selected_feature_list]
    #we will uncomment it for getting selected feature list which we will use in predict.py
    #print(selected_feature_list)
    return X

def gaussian_kernel(X1, X2, sigma=1.5):
    Gram = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.ravel()
            x2 = x2.ravel()
            Gram[i, j] = np.exp(-np.sum(np.square(x1 - x2)) / (2 * (sigma**2)))
    return Gram

def train_model(train_X,train_Y) :
    #svc = SVC(C=3.5,kernel= gaussian_kernel) #its accuracy is 72.5678%
    #Gaussian kernel: K(p, q) = exp (-||p - q||^2/(2*sigma^2)
    #RBF kernel: K(p, q) = exp(-gamma*||p - q||^2)
    svc=SVC(C=3.5,kernel='rbf',gamma='scale') #its accracy is 78.783986%
    svc.fit(train_X,train_Y)
    filename = 'MODEL_FILE.sav'
    pickle.dump(svc, open(filename, 'wb'))
    
    
if __name__=="__main__":
    X,Y=Import_data()
    X=data_processing(X,Y)
    #Uncomment it when its requirement over)
    #print(mean_null_list)
    #print(All_mini_maxi_mean_normalize_value)
    train_model(X,Y)