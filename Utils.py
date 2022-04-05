import yfinance as yf
from pyfinviz.screener import Screener
import numpy as np

def get_tickers(page=10,vectorized=True,options=None):
    if options==None:
        screener = Screener(pages=[x for x in range(1, page)])
    else:
        screener = Screener(pages=[x for x in range(1, page)],filter_options=options)
    if vectorized:
        a=np.vectorize(lambda screener,i:screener.data_frames[i].Ticker.values,signature="(),()->(j)")(screener,
                                                                                             np.delete(np.arange(0,page),1))
        list_ticker=list(a.reshape(-1))
    else:
        list_ticker=[]
        for i in range(0, page):
            if i == 1:
                pass
            else:
                list_ticker=list_ticker+list(screener.data_frames[i].Ticker.values)
    return list_ticker