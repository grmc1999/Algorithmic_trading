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
from Model_search_utils import *

import fire
from tqdm import tqdm

def main():
    Tlist=get_tickers(page=3,vectorized=True,options=[Screener.SectorOption.BASIC_MATERIALS,Screener.IndexOption.SANDP_500])
    Tlist=np.unique(np.array(Tlist)).tolist()
    data = yf.download(Tlist[:], period = "5y",interval = "1d")

    combs=[]
    for n in range(1,len(Tlist[:8])+1):
        combs=combs + (list(combinations(Tlist[:8],n)))
    combs=np.vectorize(lambda c:list(c),otypes=[object])(np.array(combs)).tolist()    
    print(len(combs))

    #C_list=list(np.arange(0.01,25.0,0.01))
    C_ranges=[]
    window_list=list(np.arange(1,10))

    #C_list=list(np.arange(0.01,10.0,1))
    #window_list=list(np.arange(1,3))

    print("creating searcher")

    MS=model_searcher(50)

    C_ranges=get_ranges(0.01,3,1,1.0)

    for C_range in tqdm(C_ranges):
        params=list(product(*[window_list,Tlist ,C_range,combs]))



        for param in tqdm(params):
            MS.model_search_fs(data,"SVC","..\Results\SVC_Search_v1\model_3.npy",*param)

if __name__=="__main__":
    fire.Fire(main)
