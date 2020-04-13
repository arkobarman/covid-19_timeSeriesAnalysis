# Functions for the 4 diagnostic plots (like R)
# Code adapted from https://towardsdatascience.com/going-from-r-to-python-linear-regression-diagnostic-plots-144d1c4aa5a
from statsmodels.nonparametric.smoothers_lowess import lowess
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
The input parameter 'results' for all functions should come from the following piece of code:
import statsmodels.formula.api as smf
model = smf.ols(formula='<dependent_variable> ~ <independent_variable>', data=<data frame>)
results = model.fit()
'''

def all4DiagnosticPlots(results):
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    residualsVsFitted(results, axes[0, 0])
    normalQ_Q(results, axes[0, 1])
    scaleLocationPlot(results, axes[1, 0])
    residualsVsLeverage(results, axes[1, 1])
    

def residualsVsFitted(results, axes=None):
    residuals = results.resid
    fitted = results.fittedvalues
    smoothed = lowess(residuals,fitted)
    top3 = abs(residuals).sort_values(ascending = False)[:3]

    if axes == None:
        plt.rcParams.update({'font.size': 16})
        plt.rcParams["figure.figsize"] = (8,7)
        fig, ax = plt.subplots()
        ax.scatter(fitted, residuals, edgecolors = 'k', facecolors = 'none')
        ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        ax.set_ylabel('Residuals')
        ax.set_xlabel('Fitted Values')
        ax.set_title('Residuals vs. Fitted')
        ax.plot([min(fitted),max(fitted)],[0,0],color = 'k',linestyle = ':', alpha = .3)

        for i in top3.index:
            ax.annotate(i,xy=(fitted[i],residuals[i]))

        plt.show()
    else:
        axes.scatter(fitted, residuals, edgecolors = 'k', facecolors = 'none')
        axes.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        axes.set_ylabel('Residuals')
        axes.set_xlabel('Fitted Values')
        axes.set_title('Residuals vs. Fitted')
        axes.plot([min(fitted),max(fitted)],[0,0],color = 'k',linestyle = ':', alpha = .3)

        for i in top3.index:
            axes.annotate(i,xy=(fitted[i],residuals[i]))
    
def normalQ_Q(results, axes=None):
    sorted_student_residuals = pd.Series(results.get_influence().resid_studentized_internal)
    sorted_student_residuals.index = results.resid.index
    sorted_student_residuals = sorted_student_residuals.sort_values(ascending = True)
    df = pd.DataFrame(sorted_student_residuals)
    df.columns = ['sorted_student_residuals']
    df['theoretical_quantiles'] = stats.probplot(df['sorted_student_residuals'], dist = 'norm', fit = False)[0]
    rankings = abs(df['sorted_student_residuals']).sort_values(ascending = False)
    top3 = rankings[:3]

    x = df['theoretical_quantiles']
    y = df['sorted_student_residuals']
    if axes == None:
        fig, ax = plt.subplots()
        ax.scatter(x,y, edgecolor = 'k',facecolor = 'none')
        ax.set_title('Normal Q-Q')
        ax.set_ylabel('Standardized Residuals')
        ax.set_xlabel('Theoretical Quantiles')
        ax.plot([np.min([x,y]),np.max([x,y])],[np.min([x,y]),np.max([x,y])], color = 'r', ls = '--')
        for val in top3.index:
            ax.annotate(val,xy=(df['theoretical_quantiles'].loc[val],df['sorted_student_residuals'].loc[val]))
        plt.show()
    else:
        axes.scatter(x,y, edgecolor = 'k',facecolor = 'none')
        axes.set_title('Normal Q-Q')
        axes.set_ylabel('Standardized Residuals')
        axes.set_xlabel('Theoretical Quantiles')
        axes.plot([np.min([x,y]),np.max([x,y])],[np.min([x,y]),np.max([x,y])], color = 'r', ls = '--')
        for val in top3.index:
            axes.annotate(val,xy=(df['theoretical_quantiles'].loc[val],df['sorted_student_residuals'].loc[val]))

def scaleLocationPlot(results, axes=None):
    fitted = results.fittedvalues
    student_residuals = results.get_influence().resid_studentized_internal
    sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
    sqrt_student_residuals.index = results.resid.index
    smoothed = lowess(sqrt_student_residuals,fitted)
    top3 = abs(sqrt_student_residuals).sort_values(ascending = False)[:3]

    if axes == None:
        fig, ax = plt.subplots()
        ax.scatter(fitted, sqrt_student_residuals, edgecolors = 'k', facecolors = 'none')
        ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        ax.set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
        ax.set_xlabel('Fitted Values')
        ax.set_title('Scale-Location')
        ax.set_ylim(0,max(sqrt_student_residuals)+0.1)
        for i in top3.index:
            ax.annotate(i,xy=(fitted[i],sqrt_student_residuals[i]))
        plt.show()
    else:
        axes.scatter(fitted, sqrt_student_residuals, edgecolors = 'k', facecolors = 'none')
        axes.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        axes.set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
        axes.set_xlabel('Fitted Values')
        axes.set_title('Scale-Location')
        axes.set_ylim(0,max(sqrt_student_residuals)+0.1)
        for i in top3.index:
            axes.annotate(i,xy=(fitted[i],sqrt_student_residuals[i]))
    
def residualsVsLeverage(results, axes=None):
    student_residuals = pd.Series(results.get_influence().resid_studentized_internal)
    student_residuals.index = results.resid.index
    df = pd.DataFrame(student_residuals)
    df.columns = ['student_residuals']
    df['leverage'] = results.get_influence().hat_matrix_diag
    smoothed = lowess(df['student_residuals'],df['leverage'])
    sorted_student_residuals = abs(df['student_residuals']).sort_values(ascending = False)
    top3 = sorted_student_residuals[:3]

    x = df['leverage']
    y = df['student_residuals']
    xpos = max(x)+max(x)*0.01  
    if axes == None:
        fig, ax = plt.subplots()
        ax.scatter(x, y, edgecolors = 'k', facecolors = 'none')
        ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        ax.set_ylabel('Studentized Residuals')
        ax.set_xlabel('Leverage')
        ax.set_title('Residuals vs. Leverage')
        ax.set_ylim(min(y)-min(y)*0.15,max(y)+max(y)*0.15)
        ax.set_xlim(-0.01,max(x)+max(x)*0.05)
        plt.tight_layout()
        for val in top3.index:
            ax.annotate(val,xy=(x.loc[val],y.loc[val]))

        cooksx = np.linspace(min(x), xpos, 50)
        p = len(results.params)
        poscooks1y = np.sqrt((p*(1-cooksx))/cooksx)
        poscooks05y = np.sqrt(0.5*(p*(1-cooksx))/cooksx)
        negcooks1y = -np.sqrt((p*(1-cooksx))/cooksx)
        negcooks05y = -np.sqrt(0.5*(p*(1-cooksx))/cooksx)

        ax.plot(cooksx,poscooks1y,label = "Cook's Distance", ls = ':', color = 'r')
        ax.plot(cooksx,poscooks05y, ls = ':', color = 'r')
        ax.plot(cooksx,negcooks1y, ls = ':', color = 'r')
        ax.plot(cooksx,negcooks05y, ls = ':', color = 'r')
        ax.plot([0,0],ax.get_ylim(), ls=":", alpha = .3, color = 'k')
        ax.plot(ax.get_xlim(), [0,0], ls=":", alpha = .3, color = 'k')
        ax.annotate('1.0', xy = (xpos, poscooks1y[-1]), color = 'r')
        ax.annotate('0.5', xy = (xpos, poscooks05y[-1]), color = 'r')
        ax.annotate('1.0', xy = (xpos, negcooks1y[-1]), color = 'r')
        ax.annotate('0.5', xy = (xpos, negcooks05y[-1]), color = 'r')
        ax.legend()
        plt.show()
    else:
        axes.scatter(x, y, edgecolors = 'k', facecolors = 'none')
        axes.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        axes.set_ylabel('Studentized Residuals')
        axes.set_xlabel('Leverage')
        axes.set_title('Residuals vs. Leverage')
        axes.set_ylim(min(y)-min(y)*0.15,max(y)+max(y)*0.15)
        axes.set_xlim(-0.01,max(x)+max(x)*0.05)
        plt.tight_layout()
        for val in top3.index:
            axes.annotate(val,xy=(x.loc[val],y.loc[val]))

        cooksx = np.linspace(min(x), xpos, 50)
        p = len(results.params)
        poscooks1y = np.sqrt((p*(1-cooksx))/cooksx)
        poscooks05y = np.sqrt(0.5*(p*(1-cooksx))/cooksx)
        negcooks1y = -np.sqrt((p*(1-cooksx))/cooksx)
        negcooks05y = -np.sqrt(0.5*(p*(1-cooksx))/cooksx)

        axes.plot(cooksx,poscooks1y,label = "Cook's Distance", ls = ':', color = 'r')
        axes.plot(cooksx,poscooks05y, ls = ':', color = 'r')
        axes.plot(cooksx,negcooks1y, ls = ':', color = 'r')
        axes.plot(cooksx,negcooks05y, ls = ':', color = 'r')
        axes.plot([0,0],axes.get_ylim(), ls=":", alpha = .3, color = 'k')
        axes.plot(axes.get_xlim(), [0,0], ls=":", alpha = .3, color = 'k')
        axes.annotate('1.0', xy = (xpos, poscooks1y[-1]), color = 'r')
        axes.annotate('0.5', xy = (xpos, poscooks05y[-1]), color = 'r')
        axes.annotate('1.0', xy = (xpos, negcooks1y[-1]), color = 'r')
        axes.annotate('0.5', xy = (xpos, negcooks05y[-1]), color = 'r')
        axes.legend()