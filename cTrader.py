
import alpaca_trade_api as tradeapi

import numpy as np
import tulipy as ti
import os, time, threading, pytz
import pandas as pd

from datetime import datetime, timezone, timedelta
from other_functions import *
from math import ceil

class Trader:
    def __init__(self, API_KEY, API_SECRET_KEY, _L, maxAttempts,account):
        self._L = _L
        self.thName = threading.currentThread().getName()

        self.maxAttempts=maxAttempts

        try:
            self.API_KEY = API_KEY
            self.API_SECRET_KEY = API_SECRET_KEY
            self.ALPACA_API_URL = "https://paper-api.alpaca.markets"
            self.alpaca = tradeapi.REST(self.API_KEY, self.API_SECRET_KEY, self.ALPACA_API_URL, api_version='v2') # or use ENV Vars

        except Exception as e:
            self._L.info('ERROR_IN: error when initializing: ' + str(e))


    def announce_order(self):
        # this function acts as a visual aid

        self._L.info('#\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t#')
        self._L.info('#\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t#')
        self._L.info('#\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t#')
        self._L.info('# O R D E R   S U B M I T T E D       ')
        self._L.info('#\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t#')
        self._L.info('#\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t#')
        self._L.info('#\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t#')

    def submitOrder(self,orderDict):
        # this is a custom function, that secures the submission
        # order dict contains the order information

        self.announce_order()

        side = orderDict['side']
        symbol = orderDict['symbol']
        qty = orderDict['qty']
        time_in_force = 'gtc'

        if orderDict['type'] is 'limit': # adjust order for a limit type
            type = 'limit'
            self._L.info('Desired limit price for limit order: %.3f$' % orderDict['limit_price'])

            if side is 'buy':
                limit_price = orderDict['limit_price'] * (1+self.pctMargin)
                # this line modifies the price that comes from the orderDict
                # adding the needed flexibility for making sure the order goes through
            elif side is 'sell':
                limit_price = orderDict['limit_price'] * (1-self.pctMargin)
            else:
                self._L.info('Side not identified: ' + str(side))
            self._L.info('Corrected (added margin) limit price: %.3f$' % limit_price)

        elif orderDict['type'] is 'market': # adjust order for a market type
            type = 'market'
            self._L.info('Desired limit price for market order: %.3f$' % orderDict['limit_price'])


        attempt = 0
        while attempt < self.maxAttempts:
            try:
                if type is 'limit':
                    self.order = self.alpaca.submit_order(
                                            side=side,
                                            qty=qty,
                                            type=type,
                                            time_in_force=time_in_force,
                                            symbol=symbol,
                                            limit_price=limit_price)

                    self._L.info("Limit order of | %d %s %s | submitted" % (qty,symbol,side))
                    self._L.info(self.order)
                    return True

                elif type is 'market':
                    self.order = self.alpaca.submit_order(
                                            side=side,
                                            qty=qty,
                                            type=type,
                                            time_in_force=time_in_force,
                                            symbol=symbol)

                    self._L.info("Market order of | %d %s %s | submitted" % (qty,symbol,side))
                    self._L.info(self.order)
                    return True

            except Exception as e:
                self._L.info('WARNING_EO: order of | %d %s %s | did not enter' % (qty,symbol,side))
                self._L.info(str(e))
                attempt += 1

        self._L.info('WARNING_SO: Could not submit the order, aborting (submitOrder)')
        return False

    def cancelOrder(self,orderId):
        # this is a custom function, that secures the cancelation

        attempt = 0
        while attempt < self.maxAttempts:
            try:
                ordersList = self.alpaca.list_orders(status='new',limit=100)

                # find the order ID and the closed status, check it matches
                for order in ordersList:
                    if order.id == orderId:
                        self._L.info('Cancelling order for ' + order.symbol)
                        self.alpaca.cancel_order(order.id)
                        return True
            except Exception as e:
                self._L.info('WARNING_CO! Failed to cancel order, trying again')
                self._L.info(e)
                self._L.info(str(ordersList))
                attempt += 1

                time.sleep(5)

        self._L.info('DANGER: order could not be cancelled, blocking thread')


    ################## RUN ##################
    def run(self,stock):
        # this is the main thread

        self._L.info('\n\n\n # #  R U N N I N G   B O T ––> (%s with %s) # #\n' % (stock.name,self.thName))

        if self.check_position(stock,maxAttempts=2): # check if the position exists beforehand
            self._L.info('There is already a position open with %s, aborting!' % stock.name)
            return stock.name,True

        if not self.is_tradable(stock.name):
            return stock.name,True

        # 1. GENERAL TREND
        if not self.get_general_trend(stock): # check the trend
            return stock.name,True

        if not self.is_tradable(stock.name,stock.direction): # can it be traded?
            return stock.name,True

        self.timeout = 0
        while True:

            #self.load_historical_data(stock,interval=gvars.fetchItval['little'])

            # 2. INSTANT TREND
            #if not self.get_instant_trend(stock):
            #    continue # restart the loop

            # 3. RSI
            #if not self.get_rsi(stock):
            #    continue # restart the loop

            # 4. STOCHASTIC
            #if not self.get_stochastic(stock,direction=stock.direction):
            #    continue # restart the loop

            currentPrice = self.get_last_price(stock)
            sharesQty = self.get_shares_from_equity(currentPrice)
            
            if not sharesQty: # if no money left...
                continue # restart the loop

            self._L.info('%s %s stock at %.3f$' % (stock.direction,stock.name,currentPrice))

            orderDict = {
                        'symbol':stock.name,
                        'qty':sharesQty,
                        'side':stock.direction,
                        'type':'limit',
                        'limit_price':currentPrice
                        }

            self._L.info('[%s] Current price read: %.2f' % (stock.name,currentPrice))

            if not self.submitOrder(orderDict): # check if the order has been SENT
                self._L.info('Could not submit order, RESTARTING SEQUENCE')
                return stock.name,False

            if not self.check_position(stock): # check if the order has EXISTS
                self._L.info('Order did not become a position, cancelling order')
                self.cancelOrder(self.order.id)
                self._L.info('Order cancelled correctly')
                return stock.name,False

            try: # go on and enter the position
                self.enter_position_mode(stock,currentPrice,sharesQty)
            except Exception as e:
                self._L.info('ERROR_EP: error when entering position')
                self._L.info(str(e))

            self._L.info('\n\n##### OPERATION COMPLETED #####\n\n')
            time.sleep(3)

            try:
                if 'YES' in self.success:
                    self._L.info(self.success)
                    return stock.name,False
                else:
                    self._L.info('Blocking asset due to bad strategy')
                    return stock.name,True
            except Exception as e:
                self._L.info('ERROR_SU: failed to identify success')
                self._L.info(str(e))