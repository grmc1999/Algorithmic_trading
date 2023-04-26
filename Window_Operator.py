from datetime import datetime
from datetime import time
from datetime import timedelta
import fire

from Investors import *





def main():

    inv=SVC_investor(
        trade_freq=timedelta(minutes=60),
        model_description_dir="../Results/SVC_HOURS/Meta_SVC.npy",
        estimation_results_directory="../Results/SVC_HOURS/Running_results",
        market_availability=[time(hour=8,minute=15,second=0),time(hour=23,minute=40,second=0)],
        API_KEY="PKE3DH2LZD3W75TOI95M",
        secret_KEY="iWdYAw2IX9VfB7fJvB2djlDHHNY8sUMtTHxtW3lk"
    )

    inv.run_invest()

if __name__=="__main__":
    fire.Fire(main)