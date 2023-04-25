# Definition of the Q-LSTM architecture for measuring the systemic risk. 


# Input: the 3-dimensional  (sample size, time lags, number of banks) array of the past log-returns R_t. 
# Output: 4 vectors containing  the VaR_50,t VaR_p,t, CoVaR_50,t, and CoVaR_p,t of the different financial institutions. 
