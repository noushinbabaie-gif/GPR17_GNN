import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

def plot_residuals_vs_predicted(preds, trues, idx, output_dir):
    residuals = trues - preds
    std_residuals = np.std(residuals)
    plt.figure(figsize=(6, 5))
    plt.scatter(preds, residuals, alpha=0.7)
    for i in range(len(preds)):
        if residuals[i]> 3*std_residuals or residuals[i]<-3*std_residuals:
            plt.annotate(idx[i], (preds[i], residuals[i]), textcoords="offset points", xytext=(5,5), ha='center', fontsize = 12)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted')
    plt.savefig(os.path.join(output_dir, "Residuals_vs_Predicted.png"), dpi=300)
    plt.show()

def normal_probability_plot(preds, trues, output_dir):
    residuals = trues - preds
    plt.figure(figsize=(6,5))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Normal Probability Plot of Residuals')
    plt.savefig(os.path.join(output_dir, "NPP.png"), dpi=300)
    plt.show()

def williams_plot(train, val, test, train_pred, val_pred, test_pred, train_nom, val_nom, test_nom, train_idx, val_idx, test_idx, output_dir):
    """
    plot all three data sets on one williams plot
    """
    #Notation
    def notation(h, h_crit, std_residuals, idx):
        for i in range(len(h)):
            if h[i]>h_crit or std_residuals[i]>3 or std_residuals[i]<-3:
                plt.annotate(idx[i], (h[i],std_residuals[i]),textcoords= "offset points", xytext= (5,5), ha='center', fontsize=12)
              

    #Calculate leverage values
    try:
        scores_dict = {
            'train' : train,
            'val' : val,
            'test' : test
            }
        train_inv = np.linalg.pinv(train.T@train)
        leverage_dict = {}
        for dset, scores in scores_dict.items():
            leverage_dict[dset] = np.diag(scores@train_inv@scores.T)
            
            
        train_lev = leverage_dict['train']
        val_lev = leverage_dict['val']
        test_lev = leverage_dict['test']
        
    except Exception as e:
        print(f'the error in leverage calculation is: {e}')
    
    #Calculate std residuals
    try:
        pred_dict = {
            'train':{'pred': train_pred, 'nom': train_nom},
            'val':{'pred':val_pred, 'nom': val_nom},
            'test':{'pred':test_pred, 'nom': test_nom}
            }

        std_residual_dict = {}  
        for name, data in pred_dict.items():
            residual = data['nom'] - data['pred']
            std_residual_dict[name] = (residual - np.mean(residual))/(np.std(residual) + 1e-9)
            
        train_std_residuals = std_residual_dict['train']
        val_std_residuals = std_residual_dict['val']
        test_std_residuals = std_residual_dict['test']
    except Exception as e:
        print(f"The error in std_residual calculation is: {e}")
        
    # Critical leverage threshold
    p = train.shape[1]  # Number of variables
    n = len(train_pred)
    h_crit = 3 * (p + 1) / n
    
    plt.figure(figsize=(10,8))
    plt.scatter(train_lev, train_std_residuals, label= 'Train', edgecolors='k', color = 'blue', s = 60)
    plt.scatter(val_lev, val_std_residuals, label= 'Validation', edgecolors= 'k', color = 'green', s = 60)
    plt.scatter(test_lev, test_std_residuals, label = 'Test', marker= 'D',edgecolors='k', color = 'orange', s = 60)
    notation(train_lev, h_crit, train_std_residuals, train_idx)
    notation(val_lev, h_crit, val_std_residuals, val_idx)
    notation(test_lev, h_crit, test_std_residuals, test_idx)
    plt.axhline(y=3, color='r' , linestyle = '--')
    plt.axhline(y=-3, color= 'r', linestyle = '--')
    plt.axvline(x = h_crit, color= 'g', linestyle = '--')
    plt.annotate(f' h* = {h_crit:0.2f}', (h_crit, np.min(train_std_residuals)), fontsize = 16)
    plt.legend(fontsize = 14, loc = 'upper right')
    plt.xlabel('Leverage', fontsize = 14)
    plt.ylabel('Standardized Residuals', fontsize = 14)
    plt.title('Williams plot', fontsize = 16)
    plt.savefig(os.path.join(output_dir,'Williams_Plot.png'), dpi=300)
    plt.show()

def plot_true_vs_predicted(
                            true_train,
                            true_val,
                            true_test,
                            pred_train,
                            pred_val,
                            pred_test,
                            idx_train,
                            idx_val,
                            idx_test,
                            output_dir):
    
    data_dict={
        'train': {'true': true_train,'pred': pred_train},
        'val': {'true': true_val, 'pred': pred_val},
        'test': {'true': true_test, 'pred': pred_test}
        }
    # Ensure inputs are numpy arrays
    try:
        data_np_dict = {}
        for name, data in data_dict.items():
            data_np_dict[name] = {'true': np.array(data['true'], dtype=float),
                                  'pred': np.array(data['pred'], dtype=float)}
    except Exception as e:
        print(e)   
        
    # Add a constant term for intercept
    x_train = data_np_dict['train']['true']
    y_train = data_np_dict['train']['pred']
    X_train = sm.add_constant(x_train)

    
    # Fit the regression model: y = intercept + slope*x
    model = sm.OLS(y_train, X_train).fit()
    
    # Extract slope, intercept, standard errors, R²
    intercept = model.params[0]
    slope = model.params[1]
    intercept_se = model.bse[0]
    slope_se = model.bse[1]
    r_squared = model.rsquared
    
    # Generate points for the regression line
    x_line = np.linspace(x_train.min(), x_train.max(), 100)
    X_line = sm.add_constant(x_line)
    predictions = model.get_prediction(X_line)
    pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI
    lower_bound_regression = pred_summary["mean_ci_lower"]
    upper_bound_regression = pred_summary["mean_ci_upper"]
    
    pred_summary_99 = predictions.summary_frame(alpha=0.01)  # 95% CI
    lower_bound_regression_99 = pred_summary_99["mean_ci_lower"]
    upper_bound_regression_99 = pred_summary_99["mean_ci_upper"]

    
    # --- Confidence Interval Around Predictions ---
    # Calculate 95% confidence interval
    residuals = y_train - x_train
    std_res = np.std(residuals)
    n = len(x_train)
    standard_error = std_res * np.sqrt(1 + 1/n + (x_train - np.mean(x_train))**2 / np.sum((x_train - np.mean(x_train))**2))
    t_val = stats.t.ppf(0.975, n - 2)  # 95% confidence
    y_fit = model.predict(X_train)
    upper_bound_prediction = y_fit + t_val * standard_error  # This should be 1D
    lower_bound_prediction = y_fit - t_val * standard_error  # This should be 1D
    
    t_val_99 = stats.t.ppf(0.999, n - 2)  # 95% confidence
    upper_bound_prediction_99 = y_fit + t_val_99 * standard_error  # This should be 1D
    lower_bound_prediction_99 = y_fit - t_val_99 * standard_error  # This should be 1D

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Fill the confidence interval around the regression line
    ax.fill_between(
        x_line, lower_bound_regression_99, upper_bound_regression_99,
        color='darkgray', alpha=0.9, label='99% CI (Regression Line)'
    )
    
    ax.fill_between(
        x_line, lower_bound_regression, upper_bound_regression,
        color='yellow', alpha=0.8, label='95% CI (Regression Line)'
    )
    

    # Fill the confidence interval
    ax.fill_between(x_train, lower_bound_prediction, upper_bound_prediction, 
                    color='red', alpha=0.2, 
                    label='95% Confidence Interval')
    
    ax.fill_between(x_train, lower_bound_prediction_99, upper_bound_prediction_99, 
                    color='purple', alpha=0.2, 
                    label='99% Confidence Interval')
    
    # Scatter plot of true vs. predicted
    ax.scatter(x_train, y_train, color='blue', alpha=0.8, label='Train Data')
    ax.scatter(data_np_dict['val']['true'], data_np_dict['val']['pred'], color='green', alpha=0.8, label='Validation Data')
    ax.scatter(data_np_dict['test']['true'], data_np_dict['test']['pred'], color='brown', alpha=0.8, label='Test Data')

    # Plot the regression line
    ax.plot(x_line, predictions.predicted_mean, color='black', linewidth=2, label='Best-fit line')
    
   
    # Define the text for the annotation
    eq_text = (
        f"y = {intercept:.3f} (±{intercept_se:.3f}) + "
        f"{slope:.3f} (±{slope_se:.3f}) × x"
        f"\nR² = {r_squared:.3f}"
    )
    
    # Place annotation inside the plot
    ax.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, backgroundcolor='white',
                verticalalignment='top')
    
    # Format the plot
    ax.set_xlabel('True pIC50', fontsize=12)
    ax.set_ylabel('Predicted pIC50', fontsize=12)
    ax.set_title('True vs. Predicted pIC50', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, "True_vs_Predicted.png"), dpi=300)
    
    # Adjust layout
    plt.tight_layout()
