# Definition of the Q-LSTM architecture for measuring the systemic risk. 


# Input: a 3-dimensional array R_t (sample size, time lags, number of banks) of the past log-returns. 
# Output: 4 vectors with VaR_50 VaR_p, CoVaR_50, and CoVaR_p of the different financial institutions. 
