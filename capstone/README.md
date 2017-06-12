# MLND Deep Learning Capstone

This project is done in the context of Udacity's Machine Learning Nano Degree
capstone project. The objective of this project is to determine whether machine
learning techniques (in particular, classifiers) can be used to predict the
directionality (i.e. **up** or **down**) of the S&P500
stock market index on any given day, based on the market opening price and an
assortment of historical market data.

## Data

The data used in this project is historical market data from January 29th, 1993
to September 29th, 2016 (at the time of writing, the python libs to access Yahoo
Finance data no longer work...). The start date was chosen based on this date
being the earliest date for which volatility (VIX) data was available via
Yahoo! Finance.

The raw data is included in the ‘SP500_historical.csv’ file provided as part of
the project submission. The raw data file contained 10 fields:

The 10 raw data fields are as follows:
- Date: calendar date for any given data row
- SP_Open: opening value (recorded at 9:30ET) for the S&P500
- SP_High: highest value on any given day for the S&P500
- SP_Low: lowest value on any given day for the S&P500
- SP_Close: closing value (recorded at 16:00ET) for the S&P500
- SP_Volume: number of shares of S&P500 components traded
- Vix_Open: opening value for the VIX index
- Vix_High: highest value on any given day for the VIX index
- Vix_Low: lowest value on any given day for the VIX index
- Vix_Close: closing value for the VIX index

## Source Code

The code is split into a couple of notebooks, where the data was fetched,
explored and the models were trained.  There is also a trading simulator
in order to see the competitive advantage of the approach over a trader.

- Stock_Data_Curation_And_Exploration.ipynb: this notebook is used to curate
and explore the dataset
- SKClassifiers.ipynb: used to train sklearn classifiers
- NN.ipynb: used to train a tensorflow-based NN
- TradingSimulation.ipynb: Simulate 1 year of trading

## Requirements

This project requires **Python 2.7** and the following Python libraries installed:

- [TensorFlow](http://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [SciPy library](http://www.scipy.org/scipylib/index.html)

To install them, go to the `code` folder and type `python install -r requirements.txt`.

Also, [iPython Notebook](http://ipython.org/notebook.html) is required to run the
jupyter notebooks.

## Run

The exploratory notebooks can be run using `Ipython Notebook`.
