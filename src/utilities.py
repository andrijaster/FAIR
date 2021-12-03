import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

from aif360.datasets import AdultDataset
from aif360.datasets import GermanDataset
from aif360.datasets import MEPSDataset19
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


def Adult_dataset(name_prot = 'sex'):
    dataset_orig = AdultDataset(protected_attribute_names=['sex'],
            privileged_classes= [['Male']],
            features_to_keep=['age', 'education-num'])
    
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    
    data, _ = dataset_orig.convert_to_dataframe()
    data.rename(columns={'income-per-year':'labels'}, inplace = True)
    data.reset_index(inplace = True, drop = True)
    sensitive = data[name_prot]
    output = dataset_orig.labels
    atribute = data.drop('labels', axis = 1, inplace = False)
    atribute.drop(name_prot, axis = 1, inplace = True)  
    return data, atribute, sensitive, output, privileged_groups, unprivileged_groups


def german_dataset_age(name_prot=['age']):
    dataset_orig = GermanDataset(
        protected_attribute_names = name_prot,
        privileged_classes=[lambda x: x >= 25],      
        features_to_drop=['personal_status', 'sex'] 
    )
    
    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]
    
    data, _ = dataset_orig.convert_to_dataframe()
    data.rename(columns={'credit':'labels'}, inplace = True)
    sensitive = data[name_prot]
    output = data['labels']
    output.replace((1,2),(0,1),inplace = True)
    atribute = data.drop('labels', axis = 1, inplace = False)
    atribute.drop(name_prot, axis = 1, inplace = True)    
    return data, atribute, sensitive, output, privileged_groups, unprivileged_groups

def german_dataset_sex(name_prot=['sex']):
    dataset_orig = GermanDataset(
        protected_attribute_names = name_prot,                                                               
        features_to_drop=['personal_status', 'age'] 
    )
    
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    
    data, _ = dataset_orig.convert_to_dataframe()
    data.rename(columns={'credit':'labels'}, inplace = True)
    sensitive = data[name_prot]
    output = data['labels']
    output.replace((1,2),(0,1),inplace = True)
    atribute = data.drop('labels', axis = 1, inplace = False)
    atribute.drop(name_prot, axis = 1, inplace = True)    
    return data, atribute, sensitive, output, privileged_groups, unprivileged_groups

def medical_dataset(name_prot = 'RACE'):
    dataset_orig = MEPSDataset19()
    privileged_groups = [{'RACE': 1}]
    unprivileged_groups = [{'RACE': 0}]
    data, _ = dataset_orig.convert_to_dataframe()
    data.reset_index(inplace = True, drop = True)
    data.rename(columns={'UTILIZATION':'labels'}, inplace = True)
    sensitive = data[name_prot]
    atribute = data.drop(name_prot, axis = 1, inplace = False) 
    atribute.drop(['labels'], axis =1, inplace =True)
    output = data['labels']
    return data, atribute, sensitive, output, privileged_groups, unprivileged_groups

def Readmission_dataset():
    folder_name = os.path.join('datasets_raw','readmission.csv')
    data = pd.read_csv(folder_name)
    data.drop(['ID','readmitDAYS'], axis = 1, inplace = True)
    data.rename(columns={'readmitBIN':'labels'}, inplace = True)
    sensitive = data['FEMALE']
    output = data['labels']
    atribute = data.drop(['labels','FEMALE'], axis = 1)
    pr_gr = [{'FEMALE': 0}]
    un_gr = [{'FEMALE': 1}]
    return data, atribute, sensitive, output, pr_gr, un_gr

def format_datasets(data, atribute, sensitive, output, out_name = "labels", sens_name = "sex", test_s = 0.15, val_s = 0.15):
    data_train, data_test_all = train_test_split(data, test_size = test_s + val_s, random_state = 30)
    data_val, data_test = train_test_split(data_test_all, test_size = test_s/(test_s + val_s), random_state = 30)

    sensitive_train = data_train[sens_name]
    sensitive_val = data_val[sens_name]
    sensitive_test = data_test[sens_name]

    output_train = data_train[out_name]
    output_val = data_val[out_name]
    output_test = data_test[out_name]

    atribute_train = data_train.drop([out_name, sens_name], axis = 1, inplace=False)
    atribute_val = data_val.drop([out_name, sens_name], axis = 1, inplace=False)
    atribute_test = data_test.drop([out_name, sens_name], axis = 1, inplace=False)

    return data_train, data_test, data_val, atribute_train, atribute_val, atribute_test, sensitive_train, sensitive_val, sensitive_test, output_train, output_val, output_test


def test(dataset_val, dataset_test,
             model, y_val, y_test, A_val, A_test, thresh,
              model_AIF, k, dataloader_val, dataloader_test, protected, unprivileged_groups, privileged_groups):
    
    protected = [protected]
    bld_val = BinaryLabelDataset(df = dataset_val, label_names = ['labels'], 
                             protected_attribute_names=protected)

    bld_test = BinaryLabelDataset(df = dataset_test, label_names = ['labels'], 
                             protected_attribute_names=protected)
                            
    if np.isin(k ,model_AIF):
        y_val_pred_prob_val = model.predict_proba(bld_val)
        A_prob_val = 0
        y_val_pred_prob_test = model.predict_proba(bld_test)
        A_prob_test = 0
    else:
        y_val_pred_prob_val, A_prob_val = model.predict_proba(dataloader_val)
        y_val_pred_prob_test, A_prob_test = model.predict_proba(dataloader_test)
        
    def metrics_form(y_val_pred_prob, y_test, A_prob, A_test, bld, dataset):

        metric_arrs = np.empty([0,8])

        if np.isin(k ,model_AIF):
            y_val_pred = (y_val_pred_prob > thresh).astype(np.float64)
        else:
            y_val_pred = (y_val_pred_prob > thresh).astype(np.float64)
            # A_pred = (A_prob > thresh).astype(np.float64)
                
        metric_arrs = np.append(metric_arrs, roc_auc_score(y_test, y_val_pred_prob))
        # print("y {}".format(roc_auc_score(y_test, y_val_pred_prob)))
        metric_arrs = np.append(metric_arrs, accuracy_score(y_test, y_val_pred))

        if np.isin(k,model_AIF):
            metric_arrs = np.append(metric_arrs, 0)
        else:
            # metric_arrs = np.append(metric_arrs, roc_auc_score(A_test, A_prob))
            metric_arrs = np.append(metric_arrs, 0)
            # print("A {}".format(roc_auc_score(A_test, A_prob)))
            
        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        
        bld2 = BinaryLabelDataset(df = dataset_pred, label_names = ['labels'], protected_attribute_names = protected)
        
        metric = ClassificationMetric(
                bld, bld2,
                unprivileged_groups = unprivileged_groups,
                privileged_groups = privileged_groups)

        metric_arrs = np.append(metric_arrs, ((metric.true_positive_rate() + metric.true_negative_rate()) / 2))
        metric_arrs = np.append(metric_arrs, np.abs(metric.average_odds_difference()))
        metric_arrs = np.append(metric_arrs, metric.disparate_impact())
        metric_arrs = np.append(metric_arrs, np.abs(metric.statistical_parity_difference()))
        metric_arrs = np.append(metric_arrs, np.abs(metric.equal_opportunity_difference()))
        
        return metric_arrs
    
    metric_val = metrics_form(y_val_pred_prob_val, y_val, A_prob_val, A_val, bld_val, dataset_val)
    metric_test = metrics_form(y_val_pred_prob_test, y_test, A_prob_test, A_test, bld_test, dataset_test)
 
    return metric_val, metric_test

class Dataset_format(Dataset):
    def __init__(self, atribute, sensitive, output):
        self.atribute = atribute.values
        self.sensitive = sensitive.values
        self.output = output.values
    
    def __len__(self):
        return len(self.atribute)

    def __getitem__(self, idx):
        return self.atribute[idx], self.output[idx], self.sensitive[idx] 

def Pareto_optimal(dataset, FAIR = True):
    
    def identify_pareto(scores):
        # Count number of items
        population_size = scores.shape[0]
        # Create a NumPy index for scores on the pareto front (zero indexed)
        population_ids = np.arange(population_size)
        # Create a starting list of items on the Pareto front
        # All items start off as being labelled as on the Parteo front
        pareto_front = np.ones(population_size, dtype=bool)
        # Loop through each item. This will then be compared with all other items
        for i in range(population_size):
            # Loop through all other items
            for j in range(population_size):
                # Check if our 'i' pint is dominated by out 'j' point
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break
        # Return ids of scenarios on pareto front
        return population_ids[pareto_front]

    points = pd.DataFrame()

    for i in dataset.index.unique():    
        score = dataset[dataset.index == i].values.copy()
        if FAIR == True:
            score[:,1] = 100 - score[:,1]
        population_ids = identify_pareto(score)
        points = points.append(dataset[dataset.index == i].iloc[population_ids,[2,3]])

    score = points.values.copy()
    if FAIR == True:
        score[:,1] = 100 - score[:,1]
    
    population_ids = identify_pareto(score)
    pareto_optimal = points.iloc[population_ids,:]

    return pareto_optimal, points

def Pareto_optimal_total(dataset, FAIR = True, name = "proba"):

    def identify_pareto(scores):
        # Count number of items
        population_size = scores.shape[0]
        # Create a NumPy index for scores on the pareto front (zero indexed)
        population_ids = np.arange(population_size)
        # Create a starting list of items on the Pareto front
        # All items start off as being labelled as on the Parteo front
        pareto_front = np.ones(population_size, dtype=bool)
        # Loop through each item. This will then be compared with all other items
        for i in range(population_size):
            # Loop through all other items
            for j in range(population_size):
                # Check if our 'i' pint is dominated by out 'j' point
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    # j dominates i. Label 'i' point as not on Pareto front
                    pareto_front[i] = 0
                    # Stop further comparisons with 'i' (no more comparisons needed)
                    break
        # Return ids of scenarios on pareto front
        return population_ids[pareto_front]

    points = pd.DataFrame()

    for i in dataset.index.unique():    
        score = dataset[dataset.index == i].values.copy()
        if FAIR == True:
            score[:,1] = 100 - score[:,1]
            score[:,2] = 100 - score[:,2]
            score[:,3] = 100 - score[:,3]
        population_ids = identify_pareto(score)
        points = points.append(dataset[dataset.index == i].iloc[population_ids,[4,5,6,7]])

    score = points.values.copy()
    if FAIR == True:
        score[:,1] = 100 - score[:,1]
        score[:,2] = 100 - score[:,2]
        score[:,3] = 100 - score[:,3]

    population_ids = identify_pareto(score)
    pareto_optimal = points.iloc[population_ids,:]

    pareto_optimal.to_excel("{}.xlsx".format(name))

    return pareto_optimal    

def plot_Pareto_fronts(PO_points_AOD, PO_points_ASPD, PO_points_AEOD, upper_bound = 0.1, lower_bound = -0.002, name = "Readmission"):

    dict_marker = {"PR":'o', "DI-NN":'v', "DI-RF":'^', "Reweighing-NN":'>', "Reweighing-RF":'<', "FAD":'8',
                    'FAD-prob':'s', "FAIR-scalar":'p', 'FAIR-betaREP':'P', "FAIR-Bernoulli":"*", "FAIR-betaSF":"h"}

    dict_color = {"PR":'b', "DI-NN":'g', "DI-RF":'r', "Reweighing-NN":'c', "Reweighing-RF":'m', "FAD":'y',
                    'FAD-prob':'k', "FAIR-scalar":'brown', 'FAIR-betaREP':'teal', "FAIR-Bernoulli":"blueviolet", "FAIR-betaSF":"crimson"}

    
    size = 100

    figure1 = plt.figure(figsize=(9, 12))

    PO_points_AOD['labels'] = PO_points_AOD.index
    ax1 = plt.subplot(311)
    for k,d in PO_points_AOD.groupby('labels'):
        if k == "FAD-prob":
            continue
        ax1.scatter(d.iloc[:,1], d.iloc[:,0], label=k, c=dict_color[k], marker = dict_marker[k], s=size)
    # ax1.set_ylim(0.5,1)
    ax1.set_xlim(lower_bound, upper_bound)
    # ax1.set_xlim(0,model1.time_control[-1])
    ax1.set_ylabel('AUC$_y$', fontweight="bold")
    ax1.set_xlabel("AOD", fontweight="bold")
    ax1.grid()
    ax1.legend(loc = 'lower right')

    PO_points_ASPD['labels'] = PO_points_ASPD.index
    ax2 = plt.subplot(312)
    for k,d in PO_points_ASPD.groupby('labels'):
        if k == "FAD-prob":
            continue
        ax2.scatter(d.iloc[:,1], d.iloc[:,0], label=k, c=dict_color[k], marker = dict_marker[k], s=size)
    # ax2.set_ylim(0.5,1)
    ax2.set_xlim(lower_bound, upper_bound)
    # ax1.set_xlim(0,model1.time_control[-1])
    ax2.set_ylabel('AUC$_y$', fontweight="bold")
    ax2.set_xlabel("ASD", fontweight="bold")
    ax2.grid()
    ax2.legend(loc = 'lower right')

    PO_points_AEOD['labels'] = PO_points_AEOD.index
    ax3 = plt.subplot(313)
    for k,d in PO_points_AEOD.groupby('labels'):
        if k == "FAD-prob":
            continue
        ax3.scatter(d.iloc[:,1], d.iloc[:,0], label=k, c=dict_color[k], marker = dict_marker[k], s=size)    
    # ax3.set_ylim(0.5,1)
    ax3.set_xlim(lower_bound, upper_bound)
    # ax1.set_xlim(0,model1.time_control[-1])
    ax3.set_ylabel('AUC$_y$', fontweight="bold")
    ax3.set_xlabel("AEOD", fontweight="bold")
    ax3.grid()
    ax3.legend(loc = 'lower right')

    plt.setp([a.get_xticklabels() for a in [ax1, ax2]], visible=False)
    
    plt.savefig('{}.png'.format(name))

def plot_AUC_Y_AUC_A(name):

    figure2 = plt.figure(figsize=(9, 8))

    points = pd.read_excel("Results/Ger_age.xls", index_col=0)
    ax1 = plt.subplot(211)
    ax1.plot(points[points.index == 'FAIR-scalar']["alpha"], points[points.index == 'FAIR-scalar'].iloc[:,0], label = "AUC$_y$")
    ax1.plot(points[points.index == 'FAIR-scalar']["alpha"], points[points.index == 'FAIR-scalar'].iloc[:,2], label = "AUC$_s$")
    plt.xscale("log")
    ax1.set_ylabel('AUC', fontweight="bold")
    ax1.set_title("German age", fontweight="bold")
    # ax1.set_xlabel("alpha", fontweight="bold")
    ax1.grid()
    ax1.set_xlim(0, 1000)
    ax1.legend()
    
    points = pd.read_excel("Results/ger_sex.xlsx", index_col=0)
    ax2 = plt.subplot(212)
    ax2.plot(points[points.index == 'FAIR-scalar']["alpha"], points[points.index == 'FAIR-scalar'].iloc[:,0], label = "AUC$_y$")
    ax2.plot(points[points.index == 'FAIR-scalar']["alpha"], points[points.index == 'FAIR-scalar'].iloc[:,2], label = "AUC$_s$")
    plt.xscale("log")
    ax2.set_ylabel('AUC', fontweight="bold")
    ax2.set_title("German sex", fontweight="bold")
    ax2.set_xlabel(r'$\alpha$', fontweight="bold")
    ax2.grid()
    ax2.set_xlim(0, 1000)
    ax2.legend()

    plt.setp([a.get_xticklabels() for a in [ax1]], visible=False)

    plt.savefig('{}.png'.format(name))



if __name__ == "__main__":

    col_AUC_y_val = 0
    col_AUC_A_val = 7

    add = 4

    aaa = np.logspace(-2, np.log10(5), num = 8)

    points = pd.read_excel("Results/readmission.xls", index_col=0)
    po_Read_AOD, po_Read_AOD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)
    po_Read_total = Pareto_optimal_total(points.iloc[:,[0, 4, 5, 6, 7, 11, 12, 13]], FAIR=True, name = "Results/Readmission_PO")

    points = pd.read_excel("Results/Adult.xls", index_col=0)
    po_Adult_AOD, po_Adult_AOD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)
    po_Adult_total = Pareto_optimal_total(points.iloc[:,[0, 4, 5, 6, 7, 11, 12, 13]], FAIR=True,  name = "Results/Adult_PO")

    points = pd.read_excel("Results/Ger_age.xls", index_col=0)
    po_Ger_age_AOD, po_Ger_age_AOD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)
    po_Ger_age_total = Pareto_optimal_total(points.iloc[:,[0, 4, 5, 6, 7, 11, 12, 13]], FAIR=True,  name = "Results/Ger_age_PO")

    points = pd.read_excel("Results/ger_sex.xlsx", index_col=0)
    po_Ger_sex_AOD, po_Ger_sex_AOD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)
    po_Ger_sex_total = Pareto_optimal_total(points.iloc[:,[0, 4, 5, 6, 7, 11, 12, 13]], FAIR=True,  name = "Results/Ger_sex_PO")

    points = pd.read_excel("Results/MEPS19.xls", index_col=0)
    po_MEPS19_AOD, po_MEPS19_AOD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)
    po_MEPS19_total = Pareto_optimal_total(points.iloc[:,[0, 4, 5, 6, 7, 11, 12, 13]], FAIR=True,  name = "Results/MEPS19_PO")

    add = 5

    points = pd.read_excel("Results/readmission.xls", index_col=0)
    po_Read_ASPD, po_Read_ASPD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)

    points = pd.read_excel("Results/Adult.xls", index_col=0)
    po_Adult_ASPD, po_Adult_ASPD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)

    points = pd.read_excel("Results/Ger_age.xls", index_col=0)
    po_Ger_age_ASPD, po_Ger_age_ASPD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)

    points = pd.read_excel("Results/ger_sex.xlsx", index_col=0)
    po_Ger_sex_ASPD, po_Ger_sex_ASPD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)

    points = pd.read_excel("Results/MEPS19.xls", index_col=0)
    po_MEPS19_ASPD, po_MEPS19_ASPD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)

    add = 6

    points = pd.read_excel("Results/readmission.xls", index_col=0)
    po_Read_AEOD, po_Read_AEOD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)

    points = pd.read_excel("Results/Adult.xls", index_col=0)
    po_Adult_AEOD, po_Adult_AEOD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)

    points = pd.read_excel("Results/Ger_age.xls", index_col=0)
    po_Ger_age_AEOD, po_Ger_age_AEOD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)

    points = pd.read_excel("Results/ger_sex.xlsx", index_col=0)
    po_Ger_sex_AEOD, po_Ger_sex_AEOD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)

    points = pd.read_excel("Results/MEPS19.xls", index_col=0)
    po_MEPS19_AEOD, po_MEPS19_AEOD_all = Pareto_optimal(points.iloc[:,[col_AUC_y_val,col_AUC_y_val+add,col_AUC_A_val,col_AUC_A_val+add]], FAIR=True)
    

    plot_Pareto_fronts(po_Read_AOD, po_Read_ASPD, po_Read_AEOD, upper_bound = 0.03, lower_bound = -0.002, name = "IMG_results/Readmission")
    plot_Pareto_fronts(po_Read_AOD_all, po_Read_ASPD_all, po_Read_AEOD_all, upper_bound = 0.03, lower_bound = -0.002, name = "IMG_results/Readmission_all")
    plot_Pareto_fronts(po_Adult_AOD, po_Adult_ASPD, po_Adult_AEOD, upper_bound = 0.2, lower_bound = -0.002, name = "IMG_results/Adult")
    plot_Pareto_fronts(po_Adult_AOD_all, po_Adult_ASPD_all, po_Adult_AEOD_all, upper_bound = 0.2, lower_bound = -0.002, name = "IMG_results/Adult_all")
    plot_Pareto_fronts(po_Ger_age_AOD, po_Ger_age_ASPD, po_Ger_age_AEOD, upper_bound = 0.11, lower_bound = -0.002, name = "IMG_results/Ger_age")
    plot_Pareto_fronts(po_Ger_age_AOD_all, po_Ger_age_ASPD_all, po_Ger_age_AEOD_all, upper_bound = 0.11, lower_bound = -0.002, name = "IMG_results/Ger_age_all")
    plot_Pareto_fronts(po_Ger_sex_AOD, po_Ger_sex_ASPD, po_Ger_sex_AEOD, upper_bound = 0.1, lower_bound = -0.002, name = "IMG_results/Ger_sex")
    plot_Pareto_fronts(po_Ger_sex_AOD_all, po_Ger_sex_ASPD_all, po_Ger_sex_AEOD_all, upper_bound = 0.1, lower_bound = -0.002, name = "IMG_results/Ger_sex_all")
    plot_Pareto_fronts(po_MEPS19_AOD, po_MEPS19_ASPD, po_MEPS19_AEOD, name = "IMG_results/MEPS19")
    plot_Pareto_fronts(po_MEPS19_AOD_all, po_MEPS19_ASPD_all, po_MEPS19_AEOD_all, name = "IMG_results/MEPS19_all")
    plot_AUC_Y_AUC_A(name = "IMG_results/AUC_y_A")

    # data, atribute, sensitive, output, privileged_groups, unprivileged_groups = Adult_dataset()
    # data, atribute, sensitive, output, privileged_groups, unprivileged_groups = german_dataset_age()
    # data, atribute, sensitive, output, privileged_groups, unprivileged_groups = german_dataset_sex()
    # data, atribute, sensitive, output, privileged_groups, unprivileged_groups = medical_dataset()
    # data, atribute, sensitive, output, privileged_groups, unprivileged_groups = Readmission_dataset()


    # data_train, data_test, data_val, atribute_train, atribute_val, atribute_test, sensitive_train, sensitive_val, sensitive_test, output_train, output_val, output_test = format_datasets(data, atribute, sensitive, output)


    # dataset_train = Dataset_format(atribute_train, sensitive_train, output_train)
    # dataset_val = Dataset_format(atribute_val, sensitive_val, output_val)
    # dataset_test = Dataset_format(atribute_test, sensitive_test, output_test)


