# StockMarketRNN
Machine Learning Project // John Kenney (jfk150030),  Matthew Brown (meb180001), Abed Mir (amm161930), Saman Laleh (sxl173130)

Original Attributes
Date,Open,High,Low,Close,Adj Close,Volume

Date indicates the date in MM/DD/YYYY format. Open, high, low, and close refer to the price of the stock at different points during the day. Open being at the open of trading for the market, and close being the end of the trading day. High and low represent the minimum and maximum values the stock price reached that day. The adjusted closing price reflects the stock's value after accounting for any corporate actions, including stock splits, dividends, and rights offerings. Volume includes the  
number of shares bought or sold for that day.

The purpose of this project is to predict the closing price for the ticker symbol SPY.
X = [Date, ]
Y = closing price for SPY

Dependencies and Modules (make sure these are installed on your computer):
pandas
numpy
matplotlib
sklearn
plotly

Once installed, run rnn.py using command:
python rnn.py

No command line arguments are required. If you are looking to edit the RNN parameters, they are located in the main function of the class at the bottom of the file.

