#tabularq_state.py
#this file contains the creation of the current state. 


import numpy as np

class TabularQState:
	#edges are the edges of the bins for the table regarding state. 
	
	def __init__(self, return_edges=None, vol_edges=None, rsi_edges=None, rate_edges=None):
		self.return_edges = return_edges or [-0.01, -0.002, 0.002, 0.01]
		self.vol_edges = vol_edges or [0.005, 0.01, 0.02]
		self.rsi_edges = rsi_edges or [30,45,55,70]
		self.rate_edges = rate_edges or [2.75, 3.75, 4.5, 5.25, 6.5]

	def _bin(self, value, edges):
		if value is None or (isinstance(value, float) and np.isnan(value)):
			return len(edges) // 2
		return int(np.digitize(value, edges))
	def build_from_row(self, row)-> tuple:
		""" 
		row: a pandas Series (one bar) that already has the rolling features

		each of the following are over 10 (the resampling size = 1 bar) bars:
		return_10 based on recent price returns (momentum)
		vol_10 is based on recent volatility
		rsi_10 is RSI (oversold/undersold) 
		"""
		return (
			self._bin(row.get("return_10"), self.return_edges),
			self._bin(row.get("vol_10"), self.vol_edges),
			self._bin(row.get("rsi_10"), self.rsi_edges),
			self._bin(row.get("ten_year_treasury"), self.rate_edges),
		)