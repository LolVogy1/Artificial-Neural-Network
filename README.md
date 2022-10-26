# Artificial-Neural-Network
This is a reupload of a project I did for my AI Methods module at university. This was implemented in Python without using any libraries, except for pandas to read excel spreadsheets.

This is an implementation of an Artificial Neural Network, more specifically a multi-layer perceptron trained using the error backpropogation algorithm.

The ANN takes various data related to weather conditions and uses it to predict the level of pan evaporation. The code is designed to train, validate and test the ANN. It also can train the network using improved methods, such as momentum, bold driver and simulated annealing

The performance of the ANN was measured using the Root Mean Squared Error after a certain number of epochs, with a lower error indicating higher accuracy. The RMSE improved from 0.02 to 0.014 at 1000 and 5000 epochs, respectively. The improved training methods also improved performance by a small amount.

The ANN had significantly better performance than LINEST, a data driven model used as a comparison, which had a RMSE of 0.256.

Overall a pretty good attempt at making a nerual network and quite a fun challenge

