from datetime import datetime
from datetime import time
from datetime import timedelta
import fire

from Investors import *





def main():

    inv=SVC_investor(
        trade_freq=timedelta(seconds=20),
        model_description_dir="../Results/SVC_HOURS/Meta_SVC.npy",
        estimation_results_directory="../Results/SVC_HOURS/Running_results",
        market_availability=[time(hour=10,minute=0,second=0),time(hour=18,minute=34,second=0)]
    )

    inv.run_invest()

if __name__=="__main__":
    fire.Fire(main)