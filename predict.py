import numpy as np
import csv
import sys
import pickle

from validate import validate

selected_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model

def replace_null_values_with_mean(X):
    mean_of_nan=np.nanmean(X,axis=0)
    index=np.where(np.isnan(X))
    X[index]=np.take(mean_of_nan,index[1])
    return X

def mean_normalize(X, column_indices):
    mini=np.min(X[:,column_indices],axis=0)
    maxi=np.max(X[:,column_indices],axis=0)
    mean=np.mean(X[:,column_indices],axis=0)
    X[:,column_indices]=(X[:,column_indices]-mean)/(maxi-mini)
    return X


def data_processing(X_test) :
    X_test=replace_null_values_with_mean(X_test)
    for i in range(X_test.shape[1]):
        X_test=mean_normalize(X_test,i)
    X_test=X_test[:,selected_features]
    return X_test
        

def predict_target_values(test_X, model):
    pred_Y=model.predict(test_X)
    return pred_Y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    test_X=data_processing(test_X)
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_svm.csv")    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_svm.csv") 