# Overview

This is going to be an evolving project as it started as an intro to Panas but slowly evolved into a crash course on machine learning.  This subject has always interested me because I like the idea of analyzing relationships and coming to conclusions of how different variables relate to each other while being as objective as possible.  I decided to start with the data set enclosed in the repository because it would provide evolution to a machine learning test with the train by learning things related to the price of the house.  

[Software Demo Video](http://youtube.link.goes.here)

# Data Analysis Results

The first question I had was which variable had the best linear relationship with Sales Price, and that seemed to be 'GrLivArea' or General Living Area.

The second question I had was were there any transformations that would apply to the Sales Price and other key variables that had relationships that could make the data have a stronger linear relationship.  Eventually, I applied a logarithmic transformation and that made the dataset follow much closer to the normal curve.  Applying these transformations to the variables that had a strong corelation should help for predicting better house prices after I figure out what transformation I need to apply to TotalBsmtSF so that the computer can draw from 3 variables rather than 2 to predict house prices in the test.csv

# Development Environment

To develop this program, I used VSCode and Python.  The libraries I used are pandas, numpy, mathplotlib, seaborn, and a few others for select commands that I needed.

# Useful Websites

* [Learn Data Science](https://www.learndatasci.com/tutorials/predicting-housing-prices-linear-regression-using-python-pandas-statsmodels/)
* [Towards Data Science](https://towardsdatascience.com/predicting-house-prices-with-linear-regression-machine-learning-from-scratch-part-ii-47a0238aeac1)
* Youtube - Various tutorials

# Future Work

* Find the transformations for TotalBsmtSF to create a linear relationship between it and SalePrice
* Once that comparison is made, compare it with the Test Data
* Be objective and look to see where I have made some oversights!
