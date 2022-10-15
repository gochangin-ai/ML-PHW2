
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall;
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# # DataProcessing(df)  
# ## remove column ( ocean_proximity ) from df_numeric




def DataProcessing(df):

    #Drop nan
    df = df.dropna()
    df_numeric = df.drop(['ocean_proximity'],axis=1)
    df_cat = df[['ocean_proximity']] 
    

    return df_numeric, df_cat


# # DataScaling(df,scalers)
# ## get encoder lists and encoding dataframe
# 
# ### parameters : df, {DataFrame} data frame that you want to scale
# ###                       scalers, {Scaler} Scalers that you want to use




def DataScaling(df, scalers):
    df_scaled_list = []
    
    for i in scalers:
        i.fit(df)
        data_scaled = i.transform(df)
        df_scaled_list.append(pd.DataFrame(data = data_scaled, columns=df.columns))
    
    return df_scaled_list
    


# #  DataEncoding(df,encoders)
# ## get scaler lists and scaling dataframe
# 
# ### parameters : df, {DataFrame} data frame that you want to scale
# ###                       scalers, {encoders} encoders that you want to use




def DataEncoding(df, encoders):
    df_encoded_list = []
    
    encoder = LabelEncoder()
    encoder.fit(df_cat.values.ravel())
    labels = encoder.transform(df_cat)
    df_encoded_list.append((pd.DataFrame(data = labels, columns=df_cat.columns)).reset_index(drop= True))
    
    df_encoded_list.append(pd.get_dummies(df_cat).reset_index(drop= True))
    
        
    return df_encoded_list, encoder.classes_
    


# #  DfConcat(df_numeric_scaled, df_cat_encoded)
# ## get encoded and scaled dataframe and concatenate with various mix
# 
# ### parameters : df_numeric_scaled, {DataFrame} data frame that scaled
# ###                       df_cat_encoded, {DataFrame} encoders that encoded




def DfConcat(df_numeric_scaled, df_cat_encoded):
    df_list = []
    
    for i in df_numeric_scaled:
        for j in df_cat_encoded:
            df_list.append(pd.concat([i,j],axis = 1))
    
    return df_list


# # clusterToIdx(clusters,ratio)
# ## make index lists for CLARANS
# 




def clusterToIdx(clusters,ratio):
    idx_list = [-1 for i in range(ratio)]
    idx = 0

    for k in clusters:
        for i in k:
            idx_list[i] = idx
        idx = idx + 1

    return idx_list


# # show_pairplot(df)
# ## make pairplot of dataframe




def show_pairplot(df):
    sns.pairplot(df, hue = 'cluster')
    plt.show()


# # cal_corr(df)
# ## calculate correlation of each columns 
# ## and return dataframe that selected




def cal_corr(df):
    corr = df.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(corr, annot=True, fmt = '.2f', cmap = 'Blues')
    plt.show()

    print("pairs with correlation coefficient is above 0.8")
    s = corr.unstack()
    s_df = pd.DataFrame(s[s<1].sort_values(ascending=False), columns=['corr']) #sort with excepting corr = 1
    s_df2 = s_df[s_df['corr']>0.8].drop_duplicates()
    print(s_df2)

    s_dict = s_df.to_dict()
    s_keys = s_dict['corr'].keys()
    s_list = list(s_keys)
    df_select = df[[s_list[0][0],s_list[0][1]]] #select two features with the strongest correlation each other

    return df_select


# # purity_score(y_true, y_pred):
# ## compute purity score 
# 
# 
# # makeplot(title, y_list, x_list):
# ## make plot for scores (elbow,  silhouette, purity ) 




def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
def makeplot(title, y_list, x_list):
    plt.plot(x_list, y_list, label =title, marker = 'o')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.show()


# 
# # AutoML (df_list, models, ratio,DBSCAN_list)
# ## train model & plot result
# 
# ### parameters : df_list, {DataFrame} data frame list that will be used
# ###                       models, {list} models that will be used
# ###                       ratio, sample size ratio for CLARANS
# ###                       DBSCAN_list, EPS and min_sample for DBSCAN parameter




def AutoML(df_list, models, ratio, DBSCAN_list):
  for i, df in enumerate(df_list):
    print(df_list_type[i])

    for j in models:
      if str(j) == "KMeans()":
        print(str(j))
        KMeanslabel_list = [-1 for i in range(0, len(df_list)*len(n_cluster))]
        idx = 0
        kmeans_distance_tmp= []
        kmeans_silhouette_tmp = []
        kmeans_purity_tmp = []
        for k in range(2, 11, 2):
          print("K = ", k)
          df_copy = df
          kmeans = KMeans(n_clusters = k, random_state = 0)
          kmeans.fit(df)
          labels = kmeans.predict(df)
         
          
          KMeanslabel_list[idx] = labels
          
          idx = idx + 1
          df['cluster'] =labels

          
          show_pairplot(df)
          
          df_select = cal_corr(df) #calculate corr and select pair with largest corr
          
          kmeans.fit(df_select)
          kmeans_distance_tmp.append(kmeans.inertia_)
          kmeans_silhouette_tmp.append(silhouette_score(df,kmeans.labels_,metric='euclidean'))
          kmeans_purity_tmp.append(purity_score(median_house_value['label'],kmeans.labels_))
          df_select['cluster'] = kmeans.predict(df_select)
          print("select two features with the strongest correlation each other")
          show_pairplot(df_select)
          
        kmeans_sumofDistance = kmeans_distance_tmp
        kmeans_silhouette = kmeans_silhouette_tmp 
        kmeans_purity = kmeans_purity_tmp 
        makeplot("KMeans_distance", kmeans_sumofDistance, n_cluster)
        makeplot("KMeans_silhouette", kmeans_silhouette, n_cluster)
        makeplot("KMeans_Purity", kmeans_purity, n_cluster)
        print("----------")
      elif str(j) == "GaussianMixture()":
        print(str(j))
        GaussianMixturelabel_list = [-1 for i in range(0, len(df_list)*len(n_cluster))]
        idx=0
        gmm_silhouette = []
        gmm_purity= []
        for k in range(2, 11, 2):
          print("K = ", k)
          gmm = GaussianMixture(n_components= k, random_state= 0)
          gmm.fit(df)
          labels = gmm.predict(df)
          gmm_silhouette.append(silhouette_score(df,labels,metric='euclidean'))
          gmm_purity.append(purity_score(median_house_value['label'], labels))
          GaussianMixturelabel_list[idx] = labels
          idx= idx+1
          df['cluster'] =labels
          show_pairplot(df)

          df_select = cal_corr(df)
          gmm.fit(df_select)
          df_select['cluster'] = gmm.predict(df_select)
          print("selcet two feature by correlation")
          show_pairplot(df_select)
        makeplot("GaussianMixture_silhouette", gmm_silhouette, n_cluster)
        makeplot("GaussianMixture_Purity", gmm_purity, n_cluster)
        print("----------")
        

      elif str(j) == "clarans()":
        print(str(j))
        Clarance_list = [-1 for i in range(0, len(df_list)*len(n_cluster))]
        idx=0
        clarans_silhouette = []
        clarans_purity = []
        for k in range(2, 11, 2):
          print("K = ", k)
          clarans_instance = clarans(df.values.tolist(), k, 6, 4).process()
          clusters = clarans_instance.get_clusters()
          labels = clusterToIdx(clusters,ratio)
          df['cluster'] = labels
          clarans_silhouette.append(silhouette_score(df, labels, metric='euclidean'))
          clarans_purity.append(purity_score(median_house_value['label'], labels))
          show_pairplot(df)

          df_select = cal_corr(df)
          clarans_instance = clarans(df_select.values.tolist(), k, 6, 4).process()
          clusters = clarans_instance.get_clusters()
          labels = clusterToIdx(clusters,ratio)
          df_select['cluster'] = labels
          print("selcet two feature by correlation")
          show_pairplot(df_select)
        makeplot("clarans_silhouette", clarans_silhouette, n_cluster)
        makeplot("clarans_Purity", clarans_purity, n_cluster)
        print("----------")

      elif str(j) == "DBSCAN()":
        print(str(j))
        DBSCAN_label_list = [-1 for i in range(0, len(df_list)*len(DBSCAN_list["eps"])*len(DBSCAN_list["min_sample"]))]
        idx=0
        dbscan_silhouette = []
        dbscan_purity = []
        for eps in DBSCAN_list["eps"]:
            max_silhouette = -2
            max_purity = -2
            for sample in DBSCAN_list["min_sample"]:
                print("eps = ", eps,"sample = ",sample)
                dbscan = DBSCAN(eps=eps,min_samples=sample)
                labels = dbscan.fit_predict(df)
                DBSCAN_label_list[idx] = labels
                df['cluster'] = labels
                show_pairplot(df)

                df_select = cal_corr(df)
                labels = dbscan.fit_predict(df_select)
                df_select['cluster'] = labels
                print("selcet two feature by correlation")
                show_pairplot(df_select)
                try:
                    current_silhouette = silhouette_score(df_select, labels, metric='euclidean')
                except:
                    current_silhouette = -5
                if max_silhouette < current_silhouette:
                    max_silhouette = current_silhouette
                current_purity = purity_score(median_house_value['label'], labels)
                if max_purity < current_purity:
                    max_purity = current_purity
            dbscan_silhouette.append(max_silhouette)
            dbscan_purity.append(max_purity) 
        dbscan_xlist = []
        for i in DBSCAN_list["eps"]:
            tmp_str = str(i)
            dbscan_xlist.append(tmp_str)
        makeplot("DBSCAN_silhouette",dbscan_silhouette , dbscan_xlist)
        makeplot("DBSCAN_purity",dbscan_purity , dbscan_xlist)
    

                    
        print("----------")

      print("==========")
        





#main



print("=== 1. Data Load & Missing Data check")
df = pd.read_csv('housing.csv')

ratio = int(len(df) / 100 * 1.5)
df = df.sample(ratio, random_state=42)
print(df)

#hyper parameter
n_cluster = list(range(2, 11, 2))
DBSCAN_list = {'eps': [0.1, 0.2, 0.5, 5, 10, 100, 1000], 'min_sample': [10, 20]}


print("=== 2. split median_house_value & labeling")
median_house_value = pd.DataFrame(df["median_house_value"])
df = df.drop(columns=["median_house_value"])

bins = list(range(14998, 500002, 48500))
median_house_value['label'] = pd.cut(median_house_value["median_house_value"], 
                                bins, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# print(median_house_value.groupby('label')['median_house_value'].apply(my_summary).unstack())

print("=== 3. drop not use data (total_bedrooms ")
dataset = df.drop(columns=["total_bedrooms"])

print("=== 4. Preprocessing")
df_numeric, df_cat = DataProcessing(dataset)

scalers = [StandardScaler(),MinMaxScaler(), RobustScaler()]
encoders = ['LabelEncoder', 'OneHotEncoder']

df_numeric_scaled = DataScaling(df_numeric, scalers)
df_cat_encoded, labels = DataEncoding(df_cat, encoders)

df_list = DfConcat(df_numeric_scaled, df_cat_encoded)

print("=== 4. Clustering & Evaluation")
df_list_type = ['SS & LE', 'SS & d', 'MMS & LE', 'MMS & d','RS & LE', 'RS & d' ]
models = ['KMeans()','GaussianMixture()','clarans()','DBSCAN()']

AutoML(df_list, models, ratio, DBSCAN_list)








