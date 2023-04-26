import yfinance as yf
from pyfinviz.screener import Screener
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Utils import get_tickers
#import get_tickers
import scipy as sc
import sympy as sp
import copy
from statsmodels.tsa.seasonal import seasonal_decompose
import os
from joblib import dump, load

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def form_hankel(data,window_size):
    hm=np.vectorize(lambda ind,ws,data:data[ind:ind+ws],signature="(),(),(i)->(j)")(np.arange(len(data)-(window_size-1)),window_size,data)
    return hm
def complete_hankel(mat,window_size):
    c=np.empty((window_size-1,window_size))
    c[:]=np.nan
    return np.vstack((c,mat))

def window_expand_data(dataframe,key,window):
    dataframe[[key+"_"+'X'"_"+str(i) for i in range(window)]]=complete_hankel(
        form_hankel(dataframe[key].values,window),
        window)
    return dataframe

def growth_trans(Y,portcentual_growth=0):
    dY=Y[1:]-Y[:-1]
    pdY=((dY/Y[:-1])>portcentual_growth)
    return np.hstack((pdY,np.nan))

def past_data_prep(data,window=3,sample_data="Open",target_ticker="CTVA",portcentual_growth=0):
    clda=data.loc[:,sample_data].loc[:,:].dropna()

    #expand for time before
    pr=clda.columns.values.tolist()
    for t in pr:
      window_expand_data(clda,t,window)
    clda.dropna()

    #binary transform
    clda[target_ticker+"_y"]=growth_trans(clda[target_ticker].values,portcentual_growth)

    #data preparation
    pr=clda.columns.values.tolist()
    selec=[target_ticker+"_y"]
    [(selec.append(k) if "_X_" in k else None) for k in pr]
    #XY split
    X_last=clda.iloc[-1]
    XY=clda.loc[:,selec].dropna()
    selec=[]
    [(selec.append(k) if "_X_" in k else None) for k in pr]
    Y=XY.loc[:,target_ticker+"_y"]
    X=XY.loc[:,selec]
    return X,Y,X_last[:-1]

def predict_RT(model_state="../Results/SVC_DAYS/SVC_CF.joblib",options=[Screener.SectorOption.BASIC_MATERIALS,Screener.IndexOption.SANDP_500],
    interval="1h",feature="Open",window=3,period="2w"):
              
    clf = load(model_state) 


    #Retrieve last data available for model
    Tlist=get_tickers(page=3,vectorized=False,options=[Screener.SectorOption.BASIC_MATERIALS,Screener.IndexOption.SANDP_500])
    Tlist=np.unique(np.array(Tlist)).tolist()
    print(Tlist)
    cdata = yf.download(Tlist[:], period = period,interval =interval)

    print(cdata)
    cc=cdata[feature].dropna().tail(window)
    print(cc)
    pr=cc.columns.values.tolist()
    for t in pr:
        window_expand_data(cc,t,window)
    cc=cc.tail(1)
    time=cc.index

    #data preparation
    pr=cc.columns.values.tolist()
    print(pr)
    selec=[]
    [(selec.append(k) if "_X_" in k else None) for k in pr]
    #XY split
    XY=cc.loc[:,selec].dropna()
    selec=[]
    [(selec.append(k) if "_X_" in k else None) for k in pr]
    #Y=XY.loc[:,target_ticker+"_y"]
    X=XY.loc[:,selec]

    #Make prediction

    Y_est=clf.predict(X)
    return Y_est,time



def parse_filters(description_format):
    filters=[]
    filters_set=description_format["options"]
    for k in filters_set.keys():
        filters.append(getattr(getattr(Screener,k),filters_set[k]))
        
    description_format["options"]=filters
    return filters