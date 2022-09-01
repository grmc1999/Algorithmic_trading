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

from datetime import datetime
from datetime import time
from datetime import timedelta
import threading
from threading import Thread
import time as t


class SVC_investor(object):
    def __init__(self,trade_freq,model_description_dir,estimation_results_directory,market_availability):
        """
        trade_freq: timedelta object
        market_availability: List of [open_time close_time]
        """
        #prepare predict_RT args
        self.trade_freq=trade_freq
        self.model_description_dir=model_description_dir
        self.model_description=np.load(self.model_description_dir,allow_pickle=True).tolist()
        self.model_params=self.model_description["model_state_params"]
        parse_filters(self.model_params)
        self.estimation_results_directory=estimation_results_directory
        
        self.market_open=market_availability[0]
        self.market_close=market_availability[1]
        
        self.estimation={
                "Y_est":None,
                "Time":[market_availability[0]]
            }
        
        
    def predict_act(self):
        Y_est,tst=predict_RT(
                **(self.model_params)
            )
        self.estimation={
                "Y_est":Y_est,
                "Time":tst
            }
        print(Y_est)
        print(tst)
        
        # MAKE OPERATION

        self.save_act_results()
    
    def save_act_results(self):
        np.save(
            os.path.join(self.estimation_results_directory,self.estimation["Time"].strftime("%m_%d_%Y_%H_%M_%S")[0]),
            self.estimation
        )
        
    def debugg_run_invest(self):
        self.operations_threads=[]
        #DEBUG
        i=0
        dt1 = datetime.now().replace(microsecond=0)
        while i!=5: # WHILE MARKET IS OPEN
            
            t.sleep(self.trade_freq.seconds/4)
            dt2=datetime.now().replace(microsecond=0)
            print(dt1)
            print(dt2)
            
    
            if (dt2-dt1)>=(self.trade_freq):
                i=i+1
                print("making operation")
                dt1=dt2
            
                operation=Thread(self.predict_act())
                operation.start()
                self.operations_threads.append(operation)
                print(self.operations_threads)
            print("keep checking")

    def run_invest(self):
        self.operations_threads=[]
        dt2 = datetime.now().replace(microsecond=0)
        last_operation_time=self.estimation["Time"][0]
        while (dt2.time()>self.market_open and dt2.time()<self.market_close):
            
            t.sleep(self.trade_freq.seconds/4)
            dt2=datetime.now().replace(microsecond=0)
            print(dt2)
            
            if (datetime.combine(datetime.today(),last_operation_time)-dt2)>=self.trade_freq:
                print("making operation")
            
                operation=Thread(self.predict_act())
                operation.start()
                
                last_operation_time=self.estimation["Time"][0].time()
                
                self.operations_threads.append(operation)
            print("keep checking")
            
        # MAKE INFORM AND RUN REVALIDATION PROTOCOL

        Tlist=get_tickers(page=3,vectorized=True,options=self.model_params["options"])
        Tlist=np.unique(np.array(Tlist)).tolist()
        data = yf.download(Tlist[:], interval = self.model_params["interval"],period = self.model_description["Validation_period"])

        clf = load(self.model_params["model_state"]) 


        X,Y=past_data_prep(data,window=self.model_params["window"],target_ticker=self.model_description["Ticker"])
        _, X_test, _, Y_test = train_test_split(X, Y>0, test_size=0.33,shuffle=False)
        te_sc=clf.score(X_test,Y_test)
        
        self.model_validation={
            "score":te_sc,
            "timespam":[data.index[0],data.index[-1]]
        }

        np.save(
            os.path.join(self.estimation_results_directory,"VALIDATION_"+self.estimation["Time"][0].strftime("%m_%d_%Y")[0]),
            self.model_validation
        )

        print("validation score test %(ts)4f"%{
                    "ts":te_sc
                    })