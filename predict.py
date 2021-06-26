import numpy as np
import csv
import sys
import pickle

from validate import validate
mean_null_list = [[ 3.29880282,  1.08748327,  4.14690669,  0.7913257 ,  0.59560898,1.59640416,  9.61547535, 11.96503081, 17.98041373,  1.22103257,51.34110915]]

All_mini_maxi_mean_normalize_value = [[1.4900000000000002, 17.55, 3.298802816901409], [0.632, 2.2380000000000004, 1.0874832746478873], [3.5140000000000007, 4.9110000000000005, 4.146906690140845], [0.5, 1.6, 0.7913257042253521], [0.5374, 1.1721, 0.5956089788732396], [1.5890770000000003, 1.6040590000000001, 1.596404161971831], [5.56, 17.66, 9.615475352112677], [9.740000000000002, 15.900000000000002, 11.965030809859156], [1.6, 79.7, 17.980413732394368], [0.863, 2.7, 1.2210325704225353], [7.1000000000000005, 318.40000000000003, 51.341109154929576]]

selected_features=[0, 1, 2, 3, 5, 6, 9, 10]

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model

def replace_null_values_with_mean(X):
    mean_of_nan=mean_null_list[0]
    index=np.where(np.isnan(X))
    X[index]=np.take(mean_of_nan,index[1])
    return X

def mean_normalize(X, column_indices):
    mini,maxi,mean=All_mini_maxi_mean_normalize_value[column_indices]
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
    #test_X_file_path = sys.argv[1]
    test_X_file_path="test_X_svm.csv"
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="test_Y_svm.csv") 