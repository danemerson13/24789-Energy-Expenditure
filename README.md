# 24789-Energy-Expenditure
Estimating Energy Expenditure from Wearable Sensors using CNN and LSTM

(c) Mikayla Schneider, Lauren Parola, Daniel Emerson
This is a course project from 24-789 Deep Learning for Engineers, at Carnegie Mellon University. 

Abstract:
Understanding human energy expenditure is essential to improving patient health. Current methods to estimate energy expenditure are not attainable or reliable for the average consumer. Our work sought to develop an alternative method for estimating energy expenditure time using wearable sensor data. After developing six different LSTM models, we determined that models were enhanced with the use of an activity label and went on to develop a CNN classifier to distinguish activity from the data. Our CNN classifier correctly predicted 178 out of 180 activities in the dataset. By running these models on a test dataset paired with five-fold cross-validation, we found that using all sensor signals with an activity label provided the best accuracy with an MSE 0.657 METs.
