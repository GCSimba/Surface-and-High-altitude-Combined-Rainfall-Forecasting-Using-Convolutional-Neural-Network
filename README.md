# Surface and High altitude Combined Rainfall Forecasting Using Convolutional Neural Network

### Abstract:

Rainfall forecasting can guide human production and life. However, the existing methods usually have a poor prediction accuracy in short-term rainfall forecasting. Machine learning methods ignore the influence of the geographical characteristics of the rainfall area. The regional characteristics of surface and high-altitude make the prediction accuracy always fluctuate in different regions. To improve the prediction accuracy of short-term rainfall forecasting, a surface and high-Altitude Combined Rainfall Forecasting model (ACRF) is proposed. First, the weighted k-means clustering method is used to select the meteorological data of the surrounding stations related to the target station. Second, the high-altitude shear value of the target station is calculated by using the meteorological factors of the surrounding stations. Third, the principal component analysis method is used to reduce dimensions of the high-altitude shear value and the surface factors. Finally, a convolutional neural network is used to forecast rainfall. We use ACRF to test 92 meteorology stations in China. The results show that ACRF is superior to existing methods in threat rating (TS) and mean square error (MSE).

### Keywords:

Rainfall Forecasting, Machine Learning, Convolutional Neural Network,  Temporal Convolutional Network

### Main Method:

<img src="https://github.com/HHUsimba/Image-Storage/blob/master/ACRFModel.png" style="zoom: 30%;" />

### Compare Models:

In order to prove that ACRF is superior to other most advanced methods, we use a variety of methods as baseline approaches for model evaluation. They are described as follows:

ECMWF: European Centre for Medium-Range Weather Forecasts. It is a European official atmospheric prediction model.

JMA: Japan Meteorological Agency numerical weather prediction model. The atmospheric prediction model proposed by Japan is also one of the methods used by China Meteorological Administration.

BPNN: Back-Propagation Neural Networks, which is a simple machine learning method.

SVM: Support Vector Machine, which is a class of generalized linear classifiers for binary data classification based on supervised learning.

MLP: Multi-layer Perceptron, which is a feed-forward artificial neural network model in machine learning method.

ARIMA: Auto-regressive Integrated Moving Average Model, which is a common statistical method for time series problems.

LSTM: Long Short-term Memory, which is an improved recurrent neural network, it can solve the problem that RNN cannot deal with long-distance dependence.

ACRF: This is the model proposed in this paper, which is based on CNN and has an altitude combined mechanism.

These models are commonly used to forecast rainfall, so we compare these methods with the proposed model. To ensure the impact of different initial data on the results, after data preprocessing, we use the same input for methods other than ARIMA and numerical mode.

### Results:

<img src="https://github.com/HHUsimba/Image-Storage/blob/master/ACRFCompare%20.png" alt="Compare" style="zoom: 67%;" />
