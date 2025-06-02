import json
import jsonpickle
from typing import Any
from collections import deque
import numpy as np
import math
import pandas as pd
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class ConversionObservation:

    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float, sugarPrice: float, sunlightIndex: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex
        

    def get_best_buy_cost(self):
        if self.route_table.empty:
            return 0.0
        return self.route_table["effective_cost"].min()


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            [],
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()

from warnings import showwarning

class Trader:
    def __init__(self) -> None:
        self.window_size = 6
        self.sunlight_history = {}  # key = minute_bin, value = list of past indices


        self.details = {"RAINFOREST_RESIN": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                             "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                             "vol_history": [ ], "moving_average": [ ], "z_score": [ ], "limit": 50},
                              "KELP": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                       "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                       "vol_history": [ ], "moving_average": [ ], "z_score": [ ], "spread_history": [], "limit": 50},
                              "SQUID_INK": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                            "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                            "vol_history": [ ], "moving_average": [ ], "z_score": [ ], "spread_history": [], "limit": 50},
                              "CROISSANTS": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                            "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                            "vol_history": [ ], "moving_average": [ ], "z_score": [ ], "limit": 250},
                              "JAMS": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                      "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                      "vol_history": [ ], "moving_average": [ ], "z_score": [ ], "limit": 350},
                              "DJEMBES": { "price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                        "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                        "vol_history": [ ], "moving_average": [ ], "z_score": [ ], "limit": 60},
                              "PICNIC_BASKET1": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                                 "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                                 "vol_history": [ ], "moving_average": [ ], "z_score": [ ], "spread_history": [], "limit": 100},
                              "PICNIC_BASKET2": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                                 "prev_position": 0, "slope_history": [ ],"spread_history": [], "fill_history": [ ],
                                                 "vol_history": [ ], "moving_average": [ ], "z_score": [ ], "limit": 100},
                              "VOLCANIC_ROCK" : {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                                 "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                                 "vol_history": [ ], "moving_average": [ ], "z_score": [ ], "base_iv_history": [], "iv_coeffs": [],
                                                 "implied_vols": {9500: [], 9750: [], 10000: [], 10250: [], 10500: []}, "spread_history": [], "limit": 400},
                              "VOLCANIC_ROCK_VOUCHER_9500": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                                 "prev_position": 0, "slope_history": [ ], "fill_history": [ ], 
                                                 "vol_history": [ ], "moving_average": [ ], "z_score": [ ], "loss_ticks": 0, "limit": 200},
                              "VOLCANIC_ROCK_VOUCHER_9750": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                                 "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                                 "vol_history": [ ], "moving_average": [ ], "loss_ticks": 0, "z_score": [ ], "limit": 200},
                              "VOLCANIC_ROCK_VOUCHER_10000": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                                 "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                                 "vol_history": [ ], "moving_average": [ ], "loss_ticks": 0, "z_score": [ ], "limit": 200},
                              "VOLCANIC_ROCK_VOUCHER_10250": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                                 "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                                 "vol_history": [ ], "moving_average": [ ], "loss_ticks": 0, "z_score": [ ], "limit": 200},
                              "VOLCANIC_ROCK_VOUCHER_10500": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                                 "prev_position": 0, "slope_history": [ ], "fill_history": [ ],
                                                 "vol_history": [ ], "moving_average": [ ], "loss_ticks": 0, "z_score": [ ], "limit": 200},
                              "MAGNIFICENT_MACARONS": {"price_history": [ ], "prev_cost_basis": 0, "prev_market_trades": [ ],
                                                 "prev_position": 0, "slope_history": [ ], "fill_history": [ ], "limit": 200}

                            }

    def pop_append(self, lst: list, value, maxlen: int) -> list:
        lst.append(value)
        if len(lst) > maxlen:
            lst.pop(0)
        return lst

    def append_to_signal(self, product: str, key: str, value, maxlen: int = 6):
        current = self.details[product][key]
        if not isinstance(current, list):
            self.details[product][key] = value  # replace directly
        else:
            self.details[product][key] = self.pop_append(current, value, maxlen)

    def cost_basis_unrealized_pnl(self,
                    product: str,
                    state: TradingState,
                    orderdepth: OrderDepth,
                    prev_position: int,
                    prev_cost_basis: float) -> tuple:

        # logic for recursive definition of cost basis
        position = state.position.get(product, 0)


        for trade in state.own_trades.get(product, []):
            if state.timestamp - 100 == trade.timestamp:
                trade_price = trade.price
                trade_quantity = trade.quantity
                denom = trade_quantity + prev_position
                if denom == 0:
                    continue  # or handle fallback case
                prev_cost_basis = (prev_position * prev_cost_basis + trade_quantity * trade_price) / denom
                prev_position += trade_quantity
        cost_basis = prev_cost_basis



        if position != 0:
            if position > 0:
                bids = sorted(orderdepth.sell_orders.keys(), reverse=True)
                if len(bids) >= 2:
                    second_best_bid = bids[1]
                else:
                    second_best_bid = bids[0]
                unrealized_pnl = position * (second_best_bid - cost_basis)
            elif position < 0:
                asks = sorted(orderdepth.buy_orders.keys(), reverse=True)
                if len(asks) >= 2:
                    second_best_ask = asks[1]
                else:
                    second_best_ask = asks[0]
                unrealized_pnl = position * (second_best_ask - cost_basis)
        else:
            unrealized_pnl = 0
        #(self, product: str, key: str, value, maxlen: int = 6)
        self.append_to_signal(product, "prev_cost_basis", cost_basis, 6)
        self.append_to_signal(product, "prev_position", position, 6)

        return (cost_basis, unrealized_pnl)


    def current_fair_value(self,
                    product: str,
                    orderdepth: OrderDepth,
                    mid_price: bool) -> float:


        weighted_bid = 0.0
        weighted_ask = 0.0

        if orderdepth.buy_orders:
            total_bid_vol = sum(orderdepth.buy_orders.values())  # sum of volumes
            if total_bid_vol != 0:
                weighted_bid = sum(p * v for p, v in orderdepth.buy_orders.items()) / total_bid_vol

        if orderdepth.sell_orders:
            total_ask_vol = sum(abs(v) for v in orderdepth.sell_orders.values())
            if total_ask_vol != 0:
                # If your ask volumes are negative, you may do sum(p * -v), or just p * abs(v):
                weighted_ask = sum(p * abs(v) for p, v in orderdepth.sell_orders.items()) / total_ask_vol

        base_mid = (weighted_bid + weighted_ask) / 2



        total_bid_vol = sum(orderdepth.buy_orders.values())
        total_ask_vol = sum(abs(volume) for volume in orderdepth.sell_orders.values())
        obi = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

        alpha = 0.3
        fair_price = base_mid + alpha * obi * (weighted_ask - weighted_bid) / 2

        if mid_price:
            return base_mid

        return fair_price
    def current_order_book_imbalance(self, od: OrderDepth) -> float:
        bid_vol = sum(od.buy_orders.values())
        ask_vol = sum(abs(v) for v in od.sell_orders.values())
        if bid_vol + ask_vol == 0:
            return 0.0
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)



    def linear_predict(self,
                       product: str,
                       price_hist: list,
                       window_size: int,
                       state: TradingState) -> float:
        # logic for linear prediction
        # take most recent 6 values for price_hist as it will be longer than wanted for linear predict
        price_hist = np.array(price_hist[-window_size:])
        if len(price_hist) < window_size:
            return self.current_fair_value(product, state.order_depths[product], False)

        x = np.arange(len(price_hist))
        y = np.array(price_hist)

        slope, intercept = np.polyfit(x, y, 1)  # Linear fit using numpy's fast polyfit
        predicted_mid_price = intercept + slope * len(price_hist)  # Predict next step
        self.append_to_signal(product, "slope_history", slope, False)

        return predicted_mid_price

    def close_product_position(self,
                          product: str,
                          close_amount: int,
                          state: TradingState,
                          orderdepth: OrderDepth,
                          result: dict[str, list[Order]]) -> None:

        if product not in result:
            result[product] = []

        position = state.position.get(product, 0)
        orders = []

        if position == 0 or close_amount == 0:
            return

        if position > 0:
            volume_to_sell = min(position, close_amount)
            sorted_bids = sorted(orderdepth.buy_orders.keys(), reverse=True)
            for bid_price in sorted_bids:
                if volume_to_sell <= 0:
                    break
                bid_volume = orderdepth.buy_orders[bid_price]
                fill_volume = min(volume_to_sell, bid_volume)
                orders.append(Order(product, bid_price, -fill_volume))
                volume_to_sell -= fill_volume

        else:
            volume_to_buy = min(abs(position), abs(close_amount))
            sorted_asks = sorted(orderdepth.sell_orders.keys())
            for ask_price in sorted_asks:
                if volume_to_buy <= 0:
                    break
                ask_volume = abs(orderdepth.sell_orders[ask_price])
                fill_volume = min(volume_to_buy, ask_volume)
                orders.append(Order(product, ask_price, fill_volume))
                volume_to_buy -= fill_volume

        result[product].extend(orders)
        return None

    def z_score(self, product: str,
                          window_size: int) -> float:
        price_hist = np.array(self.details[product]["price_history"])
        if len(price_hist) < window_size:
            z_score = 0  # Not enough data to compute
        else:
            mean = self.details[product]["moving_average"][-1]
            std = self.details[product]["vol_history"][-1]
            if std == 0:
                z_score = 0
            else:
                z_score = (price_hist[-1] - mean) / std

        self.append_to_signal(product, "z_score", z_score, 15)
        return z_score

    def norm_cdf(self, x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def bs_call_price(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * self.norm_cdf(d2)
    
    def extract_strike(self, symbol: str) -> int:
        return int(symbol.split("_")[-1])


    def bs_implied_vol(self, S, V_market, K, T, tol=1e-3, max_iter=100):
        low_vol = 0.01
        high_vol = 3.0
        for _ in range(max_iter):
            mid_vol = (low_vol + high_vol) / 2
            price = self.bs_call_price(S, K, T, mid_vol)
            if abs(price - V_market) < tol:
                return mid_vol
            if price > V_market:
                high_vol = mid_vol
            else:
                low_vol = mid_vol
        return mid_vol

    def double_down(self, product: str,
                          state: TradingState,
                          orderdepth: OrderDepth,
                          result: dict[str, list[Order]]) -> None:
        if product not in result:
            result[product] = []

        position = state.position.get(product, 0)

        best_bid = sorted(orderdepth.buy_orders.keys(), reverse=True)[0]
        best_ask = sorted(orderdepth.sell_orders.keys())[0]

        buy_prices = sorted(orderdepth.buy_orders.keys(), reverse=True)
        if len(buy_prices) > 1:
            best_bid = buy_prices[1]
            # ... use second_best_bid safely here

        sell_prices = sorted(orderdepth.sell_orders.keys())
        if len(sell_prices) > 1:
            best_ask = sell_prices[1]
            # ... use second_best_ask safely here



        if position > 0:
            result[product].append(Order(product, best_ask, position))
        elif position < 0:
            result[product].append(Order(product, best_bid, position))
        else:
            return None

    def pair_z_score(self,product1: str,
                          product2: str,
                          window_size: int) -> float:
        p1_hist = np.array(self.details[product1]["price_history"])
        p2_hist = np.array(self.details[product2]["price_history"])

        if len(p1_hist) < window_size or len(p2_hist) < window_size:
            return 0  # Not enough data to compute

        spread = p1_hist - p2_hist
        mean = np.mean(spread)
        std = np.std(spread)

        if std == 0:
            return 0
        z_score = (spread[-1] - mean) / std
        self.append_to_signal(product1, "spread_history", z_score, False)
        return z_score

    def stdev(self,
              product: str,
              price_hist: list,
              window_size: int) -> float:
        if len(price_hist) < window_size:
            vol = 0.00  # Not enough data to compute
        else:
            vol = np.std(price_hist)
        self.append_to_signal(product, "vol_history", vol, False)
        return vol

    def moving_average(self,
                        product: str,
                        price_hist: list,
                        window_size: int) -> float:
        if len(price_hist) < window_size:
            moving_average = 0.00  # Not enough data to
        else:
            moving_average = np.mean(price_hist)
        self.append_to_signal(product, "moving_average", moving_average, False)
        return moving_average

    def synthetic_fair_value_PICNIC_BASKET(self,
                                                basket: int,
                                                state: TradingState) -> float:
        croissants_od = state.order_depths["CROISSANTS"]
        jams_od       = state.order_depths["JAMS"]
        DJEMBES_od     = state.order_depths["DJEMBES"]

        if basket == 1:
            return (
                6 * self.current_fair_value("CROISSANTS", croissants_od, False)
                + 3 * self.current_fair_value("JAMS", jams_od, False)
                + self.current_fair_value("DJEMBES", DJEMBES_od, False)
            )

        elif basket == 2:
            return (
            4 * self.current_fair_value("CROISSANTS", croissants_od, False)
            + 2 * self.current_fair_value("JAMS", jams_od, False)
        )





    def long_product(self,
                          product: str,
                          add_position: int,
                          state: TradingState,
                          orderdepth: OrderDepth,
                          result: dict[str, list[Order]]) -> None:
        if product not in result:
            result[product] = []

        limit = self.details[product]["limit"]
        position = state.position.get(product, 0)
        max_long = limit - position
        ask_prices = sorted(orderdepth.sell_orders.keys())

        for ask_price in ask_prices:


            available_volume = abs(orderdepth.sell_orders[ask_price])
            fill_volume = min(add_position, max_long, available_volume)

            if fill_volume > 0:
                result[product].append(Order(product, ask_price, fill_volume))
                add_position -= fill_volume
                max_long -= fill_volume
        return None

    def short_product(self,
                            product: str,
                            sub_position: int,
                            state: TradingState,
                            orderdepth: OrderDepth,
                            result: dict[str, list[Order]]) -> None:

        if product not in result:
            result[product] = []

        limit = self.details[product]["limit"]
        position = state.position.get(product, 0)
        max_short = -limit - position
        bid_prices = sorted(orderdepth.buy_orders.keys(), reverse=True)

        for bid_price in bid_prices:

            available_volume = orderdepth.buy_orders[bid_price]
            fill_volume = min(abs(sub_position), abs(max_short), available_volume)

            if fill_volume > 0:
                result[product].append(Order(product, bid_price, -fill_volume))
                sub_position += fill_volume
                max_short += fill_volume

        return None

    def save_relevant_details(self):
        wanted_keys = ["price_history",
                      "prev_cost_basis",
                      "prev_position",
                      "prev_market_trades",
                      "spread_history"
                      "iv_fit_data", 
                      "base_iv_history"
                      ]

        filtered_details = {}

        for product, product_info in self.details.items():

            filtered_details[product] = {
                k: product_info[k] for k in wanted_keys if k in product_info
            }

        return filtered_details

    def load_details_from_traderData(self,
                                     old_data: dict):

        for product, product_info in old_data.items():

            if product in self.details:
                for key, val in product_info.items():
                    if key in self.details[product]:
                        self.details[product][key] = val

    def market_make(self,
                    product: str,
                    state: TradingState,
                    orderdepth: OrderDepth,
                    result: dict[str, list[Order]]) -> None:
        orders = []
        #state info
        current_time = state.timestamp
        best_bid = max(orderdepth.buy_orders.keys()) if orderdepth.buy_orders else None
        best_ask = min(orderdepth.sell_orders.keys()) if orderdepth.sell_orders else None
        position = state.position.get(product, 0)
        own_trades = state.own_trades.get(product, [])
        total_bid_volume = sum(orderdepth.buy_orders.values()) if orderdepth.buy_orders else 1
        total_ask_volume = sum(abs(v) for v in orderdepth.sell_orders.values()) if orderdepth.sell_orders else 1


        #self info
        fair_price = self.linear_predict(product, self.details[product]["price_history"], 20, state)
        self.append_to_signal(product, "price_history", fair_price, 6)
        slope = self.details[product]["slope_history"][-1] if self.details[product]["slope_history"] else 0

        limit = self.details[product]["limit"]
        position_ratio = position / limit
        price_hist = self.details[product]["price_history"]



        volatility = self.stdev(product, price_hist, 20)

        cost_basis, unreal_pnl = self.cost_basis_unrealized_pnl(product, state, orderdepth, self.details[product]["prev_position"], self.details[product]["prev_cost_basis"])
        max_long = limit - position
        max_short = limit + position
        order_book_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        momentum_threshold = 2.5
        spread = 1


        bid_size = min(max_long, 30)
        ask_size = min(max_short, 30)


        position_skew_strength = 1.5  # increase this for stronger unwinding
        skew = position_skew_strength * position_ratio
        fair_price -= skew * spread


        # Now quote only slightly better than the current market, if it's advantageous:
        if best_bid is not None:
            # Your bid is either your standard calculation or just 1 tick above the current best market bid, whichever is less aggressive (lower):
            bid_level_price = min(int(np.floor(fair_price - spread / 2.0)), best_bid + 2)
        else:
            bid_level_price = int(np.floor(fair_price - spread / 2.0))

        if best_ask is not None:
            # Your ask is either your standard calculation or just 1 tick below the current best market ask, whichever is less aggressive (higher):
            ask_level_price = max(int(np.ceil(fair_price + spread / 2.0)), best_ask - 2)
        else:
            ask_level_price = int(np.ceil(fair_price + spread / 2.0))


        if cost_basis is not None:
            order_book = state.order_depths[product]

            if position > 0 and best_bid and best_bid > cost_basis:
                # Exit long position by selling at the best available bid
                available_bid_volume = order_book.buy_orders.get(best_bid, 0)
                exit_qty = min(position, available_bid_volume)
                if exit_qty > 0:
                    exit_order = Order(product, best_bid, -exit_qty)  # negative because you are selling
                    orders.append(exit_order)
                    max_long -= exit_qty

            elif position < 0 and best_ask and best_ask < cost_basis:
                # Exit short position by buying at the best available ask
                available_ask_volume = abs(order_book.sell_orders.get(best_ask, 0))
                exit_qty = min(abs(position), available_ask_volume)
                if exit_qty > 0:
                    exit_order = Order(product, best_ask, exit_qty)  # positive because you are buying
                    orders.append(exit_order)
                    max_short -= exit_qty


        # Determine momentum regime explicitly
        trend_up = slope > momentum_threshold
        trend_down = slope < -momentum_threshold

        # In an upward trend, quote primarily on bid side (buy to ride upward trend)
        if trend_up:
            bid_bias = 1.0   # full size on bid side
            ask_bias = 0.2   # small hedge on ask side
        elif trend_down:
            bid_bias = 0.2   # small hedge on bid side
            ask_bias = 1.0   # full size on ask side
        else:
            bid_bias = 1.0
            ask_bias = 1.0

        # Final scaled sizes with position limits and trend bias applied
        bid_size = min(max_long, int(limit * 0.60 * bid_bias))
        ask_size = min(max_short, int(limit * 0.60 * ask_bias))

        # Place the single bid (if we have capacity and OBI suggests it):
        if bid_size > 0:
            orders.append(Order(product, bid_level_price, bid_size))
        # Place the single ask (if we have capacity and OBI suggests it):
        if ask_size > 0:
            orders.append(Order(product, ask_level_price, -ask_size))

        result[product] = orders
        return None

# =========================================================================================================================================================#

    def run(self, state: TradingState):
        result = {}
        if state.traderData:
            old_data = jsonpickle.decode(state.traderData)
            self.load_details_from_traderData(old_data)

        if "MAGNIFICENT_MACARONS" in state.observations.conversionObservations:
            obs = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
            df = pd.DataFrame([{
                "timestamp": state.timestamp,
                "transportFees": obs.transportFees,
                "exportTariff": obs.exportTariff,
                "importTariff": obs.importTariff
            }])
        # Track sunlight index per time-bin
        sunlight_obs = state.observations.conversionObservations.get("MAGNIFICENT_MACARONS")
        if sunlight_obs:
            sunlight_idx = sunlight_obs.sunlightIndex
            time_bin = (state.timestamp % 3_000_000) // 50_000  # group by 50k tick bins (~60 bins/day)

            if time_bin not in self.sunlight_history:
                self.sunlight_history[time_bin] = []
            self.sunlight_history[time_bin].append(sunlight_idx)
            if len(self.sunlight_history[time_bin]) > 30:
                self.sunlight_history[time_bin].pop(0)


        #market make on kelp and resin

        self.market_make("KELP", state, state.order_depths["KELP"], result)
        self.market_make("RAINFOREST_RESIN", state, state.order_depths["RAINFOREST_RESIN"], result)
        # === SQUID_INK Z-Score Hybrid Strategy ===
        product = "SQUID_INK"
        ink_orders = []

        ink_od = state.order_depths[product]
        ink_fair = self.current_fair_value(product, ink_od, False)
        self.append_to_signal(product, "price_history", ink_fair, 30)

        linear_price = self.linear_predict(product, self.details[product]["price_history"], 20, state)
        slope = self.details[product]["slope_history"][-1] if self.details[product]["slope_history"] else 0

        position = state.position.get(product, 0)
        limit = self.details[product]["limit"]
        max_buy = limit - position
        max_sell = limit + position

        # Calculate Z-score from recent price history
        price_hist = np.array(self.details[product]["price_history"])
        if len(price_hist) < 30:
            z = 0
        else:
            mean = np.mean(price_hist[-30:])
            std = np.std(price_hist[-30:])
            z = (ink_fair - mean) / std if std > 0 else 0

        logger.print(f"[{product}] fair={ink_fair:.1f}, linear={linear_price:.1f}, slope={slope:.2f}, z={z:.2f}, pos={position}")

        # === TREND-FOLLOWING ENTRY ===
        if slope > 1.5 and z < 0.5 and max_buy > 0:
            qty = int(min(max_buy, abs(z) * 10))
            ink_orders.append(Order(product, int(ink_fair - 1), qty))
            logger.print(f"[{product}] Trend up: BUY {qty}")

        elif slope < -1.5 and z > 0.5 and max_sell > 0:
            qty = int(min(max_sell, abs(z) * 10))
            ink_orders.append(Order(product, int(ink_fair + 1), -qty))
            logger.print(f"[{product}] Trend down: SHORT {qty}")

        # === TREND EXIT ===
        if position > 0 and (z >= 0 or slope < 0):
            ink_orders.append(Order(product, int(ink_fair), -position))
            logger.print(f"[{product}] Trend exit: CLOSE LONG")

        elif position < 0 and (z <= 0 or slope > 0):
            ink_orders.append(Order(product, int(ink_fair), -position))
            logger.print(f"[{product}] Trend exit: CLOSE SHORT")

        # === MEAN-REVERSION ENTRY ===
        if abs(slope) < 1.5:
            if z < -1 and max_buy > 0:
                qty = int(min(max_buy, abs(z) * 10))
                ink_orders.append(Order(product, int(ink_fair - 1), qty))
                logger.print(f"[{product}] Mean-revert BUY {qty}")
            elif z > 1 and max_sell > 0:
                qty = int(min(max_sell, abs(z) * 10))
                ink_orders.append(Order(product, int(ink_fair + 1), -qty))
                logger.print(f"[{product}] Mean-revert SHORT {qty}")

        # === MEAN-REVERSION EXIT ===
        if abs(z) < 0.2 and position != 0:
            ink_orders.append(Order(product, int(ink_fair), -position))
            logger.print(f"[{product}] Mean-revert exit: CLOSE {position}")

        # === Push to result ===
        for order in ink_orders:
            if order.symbol not in result:
                result[order.symbol] = []
            result[order.symbol].append(order)


                # === Refactored Basket Arbitrage Strategy (Z-Score Spread + Dynamic Sizing) ===
        baskets = {
            "PICNIC_BASKET1": {"weights": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}, "bias": 66},
            "PICNIC_BASKET2": {"weights": {"CROISSANTS": 4, "JAMS": 2}, "bias": 35}
        }

        spread_window = 30
        entry_z = 1.5
        exit_z = 0.3
        z_scale = 2.5  # size = z * z_scale
        max_size = 6

        for basket, data in baskets.items():
            if basket not in state.order_depths:
                continue

            components = data["weights"]
            bias = data["bias"]

            # Ensure all required data is available
            if not all(p in state.order_depths for p in components):
                continue

            fair = {
                p: self.current_fair_value(p, state.order_depths[p], False)
                for p in list(components) + [basket]
            }

            synthetic = sum(fair[p] * qty for p, qty in components.items())
            actual = fair[basket]
            spread = actual - synthetic - bias

            self.append_to_signal(basket, "spread_history", spread, spread_window)

            spread_hist = self.details[basket]["spread_history"]
            if len(spread_hist) < spread_window:
                continue

            spread_mean = np.mean(spread_hist)
            spread_std = np.std(spread_hist)
            spread_z = (spread - spread_mean) / spread_std if spread_std > 0 else 0

            logger.print(f"[{basket}] spread={spread:.2f}, spread_z={spread_z:.2f}")

            pos = state.position.get(basket, 0)
            size = int(min(max_size, abs(spread_z) * z_scale))

            # Entry logic
            if spread_z > entry_z:
                result.setdefault(basket, []).append(Order(basket, int(actual), -size))  # short basket
                for p, q in components.items():
                    result.setdefault(p, []).append(Order(p, int(fair[p]), q * size))     # long components
                logger.print(f"[{basket}] SHORT {size} @ spread_z={spread_z:.2f}")

            elif spread_z < -entry_z:
                result.setdefault(basket, []).append(Order(basket, int(actual), size))   # long basket
                for p, q in components.items():
                    result.setdefault(p, []).append(Order(p, int(fair[p]), -q * size))    # short components
                logger.print(f"[{basket}] LONG {size} @ spread_z={spread_z:.2f}")

            # Exit logic
            elif abs(spread_z) < exit_z and pos != 0:
                result.setdefault(basket, []).append(Order(basket, int(actual), -pos))
                for p, q in components.items():
                    result.setdefault(p, []).append(Order(p, int(fair[p]), q * -pos))
                logger.print(f"[{basket}] EXIT @ spread_z={spread_z:.2f}, pos={pos}")

        # === IV-Scaled Voucher Arbitrage Strategy ===
        voucher_symbols = [
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000",
            "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]

        iv_data = []
        voucher_orders = []

        # Rock fair price and TTE setup
        rock_od = state.order_depths.get("VOLCANIC_ROCK")
        if rock_od:
            rock_fair = self.current_fair_value("VOLCANIC_ROCK", rock_od, True)
            TTE = max(1e-4, (7_000_000 - state.timestamp) / 1_000_000)
    # Round 7 expiry assumed

            for symbol in voucher_symbols:
                if symbol not in state.order_depths:
                    continue

                od = state.order_depths[symbol]
                if not od.buy_orders or not od.sell_orders:
                    continue

                bid = max(od.buy_orders)
                ask = min(od.sell_orders)
                mid = (bid + ask) / 2
                strike = self.extract_strike(symbol)

                iv = self.bs_implied_vol(rock_fair, mid, strike, TTE)
                iv = max(0.01, min(iv, 3.0))


                moneyness = np.log(strike / rock_fair) / np.sqrt(TTE)
                iv_data.append((symbol, moneyness, iv, mid, strike, bid, ask))
                self.details["VOLCANIC_ROCK"]["iv_fit_data"] = self.pop_append(
                    self.details["VOLCANIC_ROCK"].get("iv_fit_data", []),
                    (moneyness, iv),
                    50  # feel free to tune
                )

            iv_fit_data = self.details["VOLCANIC_ROCK"]["iv_fit_data"]
            if len(iv_fit_data) >= 5:
                x = np.array([m for m, iv in iv_fit_data])
                y = np.array([iv for m, iv in iv_fit_data])


                logger.print(f"IV Data (moneyness, IV): {list(zip(x, y))}")

                coeffs = np.polyfit(x, y, 2)
                a, b, c = coeffs
                logger.print(f"Smile fit coeffs: a={a:.5f}, b={b:.5f}, c={c:.5f}")

                # Fit curve using historical data
                coeffs = np.polyfit(x, y, 2)
                a, b, c = coeffs


                base_iv = c  # fitted IV at m=0
                self.append_to_signal("VOLCANIC_ROCK", "base_iv_history", base_iv, 50)

                base_iv_hist = self.details["VOLCANIC_ROCK"]["base_iv_history"]
                if len(base_iv_hist) >= 10:
                    base_iv_mean = np.mean(base_iv_hist)
                    base_iv_std = np.std(base_iv_hist)
                    if base_iv_std > 0:
                        base_iv_z = (base_iv - base_iv_mean) / base_iv_std
                    else:
                        base_iv_z = 0.0
                else:
                    base_iv_z = 0.0


                # Now compute residuals on *current tick* iv_data
                residuals = []
                zscores = []
                for symbol, m, iv, _, _, _, _ in iv_data:
                    fitted_iv = a * m**2 + b * m + c
                    residual = iv - fitted_iv
                    residuals.append(residual)

                residuals = np.array(residuals)
                res_std = np.std(residuals)


                if res_std < 1e-6:
                    zscores = np.zeros_like(residuals)
                else:
                    zscores = (residuals - np.mean(residuals)) / res_std


                for i, (symbol, m, iv, price, strike, bid, ask) in enumerate(iv_data):
                    z = zscores[i]
                    pos = state.position.get(symbol, 0)
                        # === NEW: Cost-basis-based exit logic ===
                    cost_basis, unrealized_pnl = self.cost_basis_unrealized_pnl(
                        symbol,
                        state,
                        state.order_depths[symbol],
                        self.details[symbol]["prev_position"],
                        self.details[symbol]["prev_cost_basis"]
                    )

                    # Track loss ticks for controlled exit
                    if unrealized_pnl < -0:
                        self.details[symbol]["loss_ticks"] = self.details[symbol].get("loss_ticks", 0) + 1
                    else:
                        self.details[symbol]["loss_ticks"] = 0

                    if self.details[symbol]["loss_ticks"] >= 5:
                        logger.print(f" EXIT {symbol}: unrealized PnL = {unrealized_pnl:.2f}, cost_basis = {cost_basis:.2f}")
                        self.close_product_position(
                            symbol,
                            abs(pos),
                            state,
                            state.order_depths[symbol],
                            result
                        )
                        self.details[symbol]["loss_ticks"] = 0  # reset

                    logger.print(f"[{symbol}] z={z:.2f}, IV={iv:.4f}, m={m:.4f}, strike={strike}, pos={pos}")
                                        # Constants for scaling
                    base_size = 20          # minimum starting size
                    z_multiplier = 40       # z-score to size scaling
                    max_order_size = 100    # cap for any one order

                    # Current position and risk limits
                    limit = self.details[symbol]["limit"]
                    pos = state.position.get(symbol, 0)
                    remaining_capacity = max(0, limit - abs(pos))

                    # Size based on signal
                    size = int(base_size + z_multiplier * abs(z))
                    size = min(size, max_order_size, remaining_capacity)

                    # Safety filters before buying
                    hist = self.details[symbol]["price_history"]
                    if len(hist) >= 5 and (hist[-1] < hist[-2] < hist[-3]):
                        continue  # falling knife

                    if ask - bid > 8 or abs(state.order_depths[symbol].sell_orders[ask]) < 5:
                        continue  # illiquid or wide

                    slope_at_m = 2 * a * m + b
                    if z < 0 and slope_at_m < -0.05:
                        continue  # IV curve descending here, maybe normal

                    if res_std > 0.15:
                        continue  # smile curve unreliable

                    if abs(base_iv_z) < 1.0:
                        continue  # market not committed enough




                    max_m = 0.25 + 0.5 * TTE
                    # === ENTRY FILTERS FOR VOUCHERS ===
                    should_enter = False

                    # --- Basic entry: market-wide sentiment must support it ---
                    if base_iv_z > 1 and z > 1 and abs(m) < max_m:
                        direction = -1  # SELL overvalued
                        entry_price = bid
                        should_enter = True
                    elif base_iv_z < -1 and z < -1 and abs(m) < max_m:
                        direction = 1  # BUY undervalued
                        entry_price = ask
                        should_enter = True

                    # --- Layered filter: reject bad entries ---
                    if should_enter:

                        # Safety 1: Skip if smile curve too noisy (bad fit)
                        if res_std > 0.15:
                            should_enter = False

                        # Safety 2: Avoid illiquid markets
                        spread_width = ask - bid
                        if spread_width > 8 or abs(state.order_depths[symbol].sell_orders.get(ask, 0)) < 5:
                            should_enter = False

                        # Safety 3: Avoid falling knives (last 4 prices falling)
                        hist = self.details[symbol]["price_history"]
                        if len(hist) >= 4 and hist[-1] < hist[-2] < hist[-3] < hist[-4]:
                            should_enter = False

                        # Safety 4: Only trade when smile slope indicates reversal
                        slope_at_m = 2 * a * m + b
                        if direction == 1 and slope_at_m < -0.03:
                            should_enter = False
                        elif direction == -1 and slope_at_m > 0.03:
                            should_enter = False

                    # === Final execution ===
                    if should_enter:
                        order = Order(symbol, entry_price, direction * size)
                        voucher_orders.append(order)
                        logger.print(f"[{symbol}] ENTER {'BUY' if direction == 1 else 'SELL'} {size} @ {entry_price} | z={z:.2f}, baseIVz={base_iv_z:.2f}, slope={slope_at_m:.3f}")



        # Push orders to result
        for order in voucher_orders:
            if order.symbol not in result:
                result[order.symbol] = []
            result[order.symbol].append(order)

        # === VOLCANIC_ROCK Z-Score Hybrid Strategy ===
        product = "VOLCANIC_ROCK"
        rock_orders = []

        rock_od = state.order_depths[product]
        rock_fair = self.current_fair_value(product, rock_od, False)
        self.append_to_signal(product, "price_history", rock_fair, 30)

        linear_price = self.linear_predict(product, self.details[product]["price_history"], 20, state)
        slope = self.details[product]["slope_history"][-1] if self.details[product]["slope_history"] else 0

        position = state.position.get(product, 0)
        limit = self.details[product]["limit"]
        max_buy = limit - position
        max_sell = limit + position

        price_hist = np.array(self.details[product]["price_history"])
        if len(price_hist) < 30:
            z = 0
        else:
            mean = np.mean(price_hist[-30:])
            std = np.std(price_hist[-30:])
            z = (rock_fair - mean) / std if std > 0 else 0

        logger.print(f"[{product}] fair={rock_fair:.1f}, linear={linear_price:.1f}, slope={slope:.2f}, z={z:.2f}, pos={position}")

        # === TREND ENTRY ===
        if slope > 1.5 and z < 0.5 and max_buy > 0:
            qty = int(min(max_buy, abs(z) * 10))
            rock_orders.append(Order(product, int(rock_fair - 1), qty))
            logger.print(f"[{product}] Trend up: BUY {qty}")

        elif slope < -1.5 and z > 0.5 and max_sell > 0:
            qty = int(min(max_sell, abs(z) * 10))
            rock_orders.append(Order(product, int(rock_fair + 1), -qty))
            logger.print(f"[{product}] Trend down: SHORT {qty}")

        # === TREND EXIT ===
        if position > 0 and (z >= 0 or slope < 0):
            rock_orders.append(Order(product, int(rock_fair), -position))
            logger.print(f"[{product}] Trend exit: CLOSE LONG")

        elif position < 0 and (z <= 0 or slope > 0):
            rock_orders.append(Order(product, int(rock_fair), -position))
            logger.print(f"[{product}] Trend exit: CLOSE SHORT")

        # === MEAN REVERSION ENTRY ===
        if abs(slope) < 1.5:
            if z < -1 and max_buy > 0:
                qty = int(min(max_buy, abs(z) * 10))
                rock_orders.append(Order(product, int(rock_fair - 1), qty))
                logger.print(f"[{product}] Mean-revert BUY {qty}")
            elif z > 1 and max_sell > 0:
                qty = int(min(max_sell, abs(z) * 10))
                rock_orders.append(Order(product, int(rock_fair + 1), -qty))
                logger.print(f"[{product}] Mean-revert SHORT {qty}")

        # === MEAN REVERSION EXIT ===
        if abs(z) < 0.2 and position != 0:
            rock_orders.append(Order(product, int(rock_fair), -position))
            logger.print(f"[{product}] Mean-revert exit: CLOSE {position}")

        # === Push to result ===
        for order in rock_orders:
            if order.symbol not in result:
                result[order.symbol] = []
            result[order.symbol].append(order)


        product = "MAGNIFICENT_MACARONS"
        macaron_od = state.order_depths[product]
        macaron_pos = state.position.get(product, 0)
        macaron_fair = self.current_fair_value(product, macaron_od, True)
        macaron_orders = []

        cycle_time = state.timestamp % 3_000_000
        limit = self.details[product]["limit"]

        # Sorted prices
        sorted_bids = sorted(macaron_od.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(macaron_od.sell_orders.keys())

        # Check if current sunlight is unusually low or high for this bin
        z_score_sunlight = 0
        if time_bin in self.sunlight_history and len(self.sunlight_history[time_bin]) >= 10:
            sun_vals = np.array(self.sunlight_history[time_bin])
            mean_sun = np.mean(sun_vals)
            std_sun = np.std(sun_vals)
            if std_sun > 0:
                z_score_sunlight = (sunlight_idx - mean_sun) / std_sun

        if z_score_sunlight < -1.0 and macaron_pos < limit:
            # Sunlight is much lower than normal ⇒ LONG
            scale = min(limit - macaron_pos, int(10 + 15 * abs(z_score_sunlight)))  # Base 10, grows with |Z|
            for ask in sorted_asks:
                available = abs(macaron_od.sell_orders[ask])
                qty = min(scale, available)
                if qty > 0:
                    macaron_orders.append(Order(product, ask, qty))
                    macaron_pos += qty
                    logger.print(f"[{product}] LOW sunlight Z={z_score_sunlight:.2f} → LONG {qty} @ {ask}")
                    break

        elif z_score_sunlight > 1.0 and macaron_pos > 0:
            # Sunlight is much higher than normal ⇒ SHORT
            scale = min(macaron_pos, int(10 + 15 * abs(z_score_sunlight)))
            for bid in sorted_bids:
                available = macaron_od.buy_orders[bid]
                qty = min(scale, available)
                if qty > 0:
                    macaron_orders.append(Order(product, bid, -qty))
                    macaron_pos -= qty
                    logger.print(f"[{product}] HIGH sunlight Z={z_score_sunlight:.2f} → SHORT {qty} @ {bid}")
                    break

        # Push to result
        if macaron_orders:
            if product not in result:
                result[product] = []
            result[product].extend(macaron_orders)

        
        # === Intelligent Conversion Strategy (Import + Export, Sunlight-aware) ===
        product = "MAGNIFICENT_MACARONS"
        conversions = 0
        conversion_limit = 10
        max_pos = self.details[product]["limit"]
        pos = state.position.get(product, 0)

        obs = state.observations.conversionObservations[product]
        od = state.order_depths[product]
        mid_price = self.current_fair_value(product, od, True)

        # External quote (Pristine Cuisine)
        import_cost = obs.askPrice + obs.transportFees + obs.importTariff
        export_value = obs.bidPrice - obs.transportFees - obs.exportTariff

        # Time bin and sunlight z-score
        sunlight_idx = obs.sunlightIndex
        time_bin = (state.timestamp % 3_000_000) // 50_000
        z_score_sunlight = 0
        if time_bin in self.sunlight_history and len(self.sunlight_history[time_bin]) >= 10:
            sun_vals = np.array(self.sunlight_history[time_bin])
            mean_sun = np.mean(sun_vals)
            std_sun = np.std(sun_vals)
            if std_sun > 0:
                z_score_sunlight = (sunlight_idx - mean_sun) / std_sun

        # === Export if Pristine offers more than market (with room) ===
        export_spread = export_value - mid_price
        if export_spread > 0.05 and pos > 0:
            qty = min(pos, conversion_limit)
            conversions = qty
            logger.print(f"[{product}] Exporting {qty} | Export={export_value:.2f} > Market={mid_price:.2f} | z_sun={z_score_sunlight:.2f}")

        # === Import if Pristine is cheaper than market AND we want to stock up ===
        elif import_cost < mid_price - 0.05 and pos < max_pos:
            # Avoid importing in long sunlit periods (storage cost adds up)
            if z_score_sunlight < 1.0:
                qty = min(conversion_limit, max_pos - pos)
                conversions = -qty
                logger.print(f"[{product}] Importing {qty} | Import={import_cost:.2f} < Market={mid_price:.2f} | z_sun={z_score_sunlight:.2f}")

        else:
            logger.print(f"[{product}] No conversion | Import={import_cost:.2f}, Export={export_value:.2f}, Market={mid_price:.2f}, Pos={pos}, z_sun={z_score_sunlight:.2f}")

        traderData = self.save_relevant_details()

        traderDataEncoded = jsonpickle.encode(traderData)
        logger.flush(state, result, conversions, traderDataEncoded)
        return result, conversions, traderDataEncoded