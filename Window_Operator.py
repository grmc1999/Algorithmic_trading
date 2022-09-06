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
        market_availability=[time(hour=8,minute=35,second=0),time(hour=14,minute=45,second=0)],
        API_KEY="PKRMQ77B325NJDDG614A",
        secret_KEY="zhQt9G6UBZ97VmedbQaBfpDxnSGAETJR9EVyCEmT"
    )

    inv.run_invest()

if __name__=="__main__":
    fire.Fire(main)