# Passenger_load_prediction_Python

This is the final project for my Python class at Ecole Polytechnique (MSc Data Science, 2019-2020).
I worked with a classmate [Tommy Tran](https://github.com/TommyTranX) to forecast airplane passengers load in the United States using machine learning in Python.

We were initially provided with:
* training data : information about US domestic flights between 2011 and 2013
* external data : meteorological information about US airports

Based on these first pieces of information, we identified several steps to work on and improve our predictions:

1. Feature engineering (resulting in our `feature_extractor.py`):\
→ look for additional external data\
→ clean it and merge it with preexisting data (into a single `external_data.csv` file)\
→ adjust data encoding\
→ check features relevance

2. Model selection and tuning (resulting in our `regressor.py'):\
→ try several models and select the best-performing\
→ try model averaging and stacking\
→ tune model hyperparameters with gridsearch

In this repository, you will find our final report and final productions (external dataset, feature extractor, regressor) as well as our working material.
In the report, we go through the different steps of the project and present you the ideas we got, the techniques we tried, the models we explored and the results we obtained.
