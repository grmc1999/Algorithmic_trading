import yfinance as yf
from pyfinviz.screener import Screener
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Utils import get_tickers
from window_time_utils import *
#import get_tickers
import scipy as sc
import sympy as sp
import copy
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from itertools import product,combinations

def get_ranges(min,max,n_ranges,c_step):
  ranges=np.arange(min,max+max/n_ranges,max/n_ranges)
  ranges=np.vstack((ranges[:-1],ranges[1:])).T
  return np.vectorize(lambda range: np.arange(range[0],range[1],c_step).tolist(),signature="(j)->()",otypes=[object])(ranges)

class model_searcher(object):
    def __init__(self,rank_size):
        self.rank={}
        for i in range(rank_size):
            self.rank[i]={"test_score":0,
                         "target_ticker":None}
        
    def model_search(self,data,Model_type,model_setting_dir,window,target_ticker,C):
        X,Y=past_data_prep(data,window,target_ticker=target_ticker)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y>0, test_size=0.33,shuffle=False)
        clf=make_pipeline(StandardScaler(),SVC(kernel="linear",C=C))
        clf.fit(X_train,Y_train)
        tr_sc=clf.score(X_train,Y_train)
        te_sc=clf.score(X_test,Y_test)
        results={
        "Model_type":Model_type,
        "train_score":tr_sc,
        "test_score":te_sc,
        "window":window,
        "target_ticker":target_ticker,
        "input_tickers":list(data["Close"].keys()),
        "C":C
        }
        top_scores=np.vectorize(lambda r:self.rank[r]["test_score"])(np.array(list(self.rank)))
        
        for i in range(len(top_scores)):
            if top_scores[i]<results["test_score"]:
                bt=np.vectorize(lambda k:self.rank[k]["target_ticker"])(list(self.rank.keys()))
                if not (results["target_ticker"] == bt[:(i)]).any():
                    self.rank[i]=results
                
                break
                
        np.save(model_setting_dir,self.rank)

    def model_search_fs(self,data,Model_type,model_setting_dir,window,target_ticker,C,feature_tickers):
        
        tickers=np.array(list(data["Close"].keys()))
        feature_tickers.append(target_ticker)
        sel_tickers=np.unique(np.array(feature_tickers))
        conditions=np.array(['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'])
        data=data[list(product(*[conditions,sel_tickers.tolist()]))]
        
        X,Y=past_data_prep(data,window,target_ticker=target_ticker)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y>0, test_size=0.33,shuffle=False)
        clf=make_pipeline(StandardScaler(),SVC(kernel="linear",C=C))
        clf.fit(X_train,Y_train)
        tr_sc=clf.score(X_train,Y_train)
        te_sc=clf.score(X_test,Y_test)
        results={
        "Model_type":Model_type,
        "train_score":tr_sc,
        "test_score":te_sc,
        "window":window,
        "target_ticker":target_ticker,
        "input_tickers":sel_tickers.tolist(),
        "C":C
        }
        top_scores=np.vectorize(lambda r:self.rank[r]["test_score"])(np.array(list(self.rank)))
        
        for i in range(len(top_scores)):
            if top_scores[i]<results["test_score"]:
                bt=np.vectorize(lambda k:self.rank[k]["target_ticker"])(list(self.rank.keys()))
                if not (results["target_ticker"] == bt[:(i)]).any():
                    self.rank[i]=results
                
                break
                
        np.save(model_setting_dir,self.rank)