#allocator.py

class RankedAllocator:
	def allocate(self, signals,scores, portfolio, prices):
		"""
		singals: dict[ticker, "buy"|"sell"|"hold"]
		portfolio: current holdings and cash
		prices: dict[ticker, price]
		returns: dict[ticker, {"size": float, "local_pnl": float}]
				plus "profolio_return": float
		"""
		buys = sorted(
			[t for t, s in signals.items() if s == "buy"],
			key=lambda t:scores.get(t,0.0), reverse=True
		)
		sells = sorted(
			[t for t, s in signals.items() if s == "sell"],
			key=lambda t:scores.get(t,0.0), reverse=True
		)

		

		#sells. highest rank sells 100% lowest rank sells 25%
		n_sell = len(sells)
		for i, ticker in enumerate(sells):
			fraction = 1.0-.75 * (i/max(len(sells)-1, 1))
			shares = portfolio.get_total_shares(ticker)*fraction
			if shares > 0:
				portfolio.sell(None, ticker, shares, prices[ticker])
				

		#buys. spends half of remaining cash on highest rank buy each iteration
		
		for ticker in buys: 
			cash = portfolio.cash
			spend = cash * .5
			if spend <= 0 or prices[ticker] <=0:
				continue
			qty = spend / prices[ticker]
			if portfolio.buy(None, ticker, qty, prices[ticker]):
				cash = portfolio.cash

		# build results
		results = {"portfolio_return" : 0.0}
		# caller will fill value_before/after and local_pnll after the fact
		for ticker in signals:
			results[ticker]={
				"turnover": 1.0 if signals[ticker] in ("buy","sell") else 0.0
			}
		return results

def attribute_rewards(
	portfolio, 
	prices_before, 
	prices_after, 
	value_before, 
	agents, 
	signals, 
	results, 
	reward_fn,
	trades_before:int, 
):
	value_after = portfolio.get_current_value(prices_after)
	portfolio_return = (value_after - value_before)/value_before if value_before > 0 else 0.0
	results["portfolio_return"] = portfolio_return

	#realized PnL
	new_trades = portfolio.trade_history[trades_before:]
	realized_by_ticker = {}
	for trade in new_trades:
		if trade.buy_sell != "sell":
			continue
		realized_by_ticker[trade.ticker]= realized_by_ticker.get(trade.ticker,0.0)+trade.realized_pnl
		
	n_agents = max(len(agents), 1)
	shared = portfolio_return/n_agents

	local_rewards ={}
	for ticker, agent in agents.items():
		mtm = portfolio.ticker_mtm(ticker, prices_before.get(ticker, 0.0), prices_after.get(ticker, 0.0))

		realized = realized_by_ticker.get(ticker, 0.0)
		turnover = results.get(ticker,{}).get("turnover",0.0)

		local = reward_fn.compute(
			value_before=value_before,
			value_after=value_before + mtm,
			realized_pnl = realized, turnover=turnover,
		)
		local_rewards[ticker]=local + shared
	return local_rewards, portfolio_return
		