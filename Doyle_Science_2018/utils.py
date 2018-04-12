import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Import relevant scikit-learn modules
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# Import rpy2 to use R
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as robjects
pandas2ri.activate()
readRDS = robjects.r['readRDS']

def fit_models(X_train, 
               X_test,
               y_train, 
               y_test,
               models=[]):
    predictions = []
    r2_values = []
    rmse_values = []
    for model in models:
        print(model)
        # fit the model and generate predictions
        model.fit(X_train, y_train.ravel())
        preds = model.predict(X_test)

        # calculate an R-squared and RMSE values
        r_squared = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        # append all to lists
        predictions.append(preds)
        r2_values.append(r_squared)
        rmse_values.append(rmse)
    print('Done fitting models')
    return predictions, r2_values, rmse_values

def plot_models(predictions,
                r2_values,
                rmse_values,
                y_test,
                titles =['Linear Regression',
                          'k-Nearest Neighbors',
                          'Support Vector Machine',
                          'Neural Network [5 neurons]',
                          'Neural Network [100 neurons]',
                          'Random Forest'],
                positions=[231,232,233,234,235,236],
                save=False):

    fig = plt.figure(figsize=(15,10))
    for pos, pred, r2, rmse, title in zip(positions,
                                          predictions,
                                          r2_values,
                                          rmse_values,
                                          titles):
        # create subplot
        plt.subplot(pos)
        plt.grid(alpha=0.2)
        plt.title(title, fontsize=15)
        
        # add score patches
        r2_patch = mpatches.Patch(label="R2 = {:04.2f}".format(r2))
        rmse_patch = mpatches.Patch(label="RMSE = {:04.1f}".format(rmse))
        plt.xlim(-25,105)
        plt.ylim(0,105)
        plt.scatter(pred, y_test, alpha=0.2)
        plt.legend(handles=[r2_patch, rmse_patch], fontsize=12)
        plt.plot(np.arange(100), np.arange(100), ls="--", c=".3")
        fig.text(0.5, 0.08, 'predicted yield', ha='center', va='center', fontsize=15)
        fig.text(0.09, 0.5, 'observed yield', ha='center', va='center', rotation='vertical', fontsize=15)
    if save:
        plt.savefig(save, dpi = 300)
    plt.show()