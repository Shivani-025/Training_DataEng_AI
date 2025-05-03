from statistics import mean
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

def test_data(predict_xvalue, m, b):
    predict_yvalue = (m * predict_xvalue) + b
    print('Test Data for x :     ', predict_xvalue, '    ', 'Test Data for y :     ', predict_yvalue)
    plt.title('Test Value')
   # plt.scatter(predict_xvalue, predict_yvalue, color='#003F72', label='data')
    plt.scatter(predict_xvalue, predict_yvalue, color='#ff0000', label='Predicted Value')
    plt.legend(loc='best')
    plt.show()


def validate_results(m,b):
    predict_xvalues = np.array([2.5, 3.5, 4.5, 5.5, 6.5], dtype=np.float64)
    predict_yvalues = [(m * x) + b for x in predict_xvalues]
    print('Validation Data Set')
    print('X values', predict_xvalues)
    print('Y values', predict_yvalues)



model_folder = '../model'
if not os.path.exists(model_folder):
    print('Model Folder does not exists.')

    #os.makedirs(model_folder)

#Save the model in the 'model' folder
model_path = os.path.join(model_folder, 'linear_regression_model.pkl')
#print(model_path)
#model = {'slope': m, 'intercept':b}
with open(model_path, 'rb') as file:
    model = pickle.load(file)

m=model['slope']
b=model['intercept']
test_data(7, m, b)
validate_results(m,b)


#===============OUTPUT=====================
'''
Test Data for x :      7      Test Data for y :      74.0
Validation Data Set
X values [2.5 3.5 4.5 5.5 6.5]
Y values [29.0, 39.0, 49.0, 59.0, 69.0]
'''
