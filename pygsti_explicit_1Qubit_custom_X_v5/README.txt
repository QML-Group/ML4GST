1. changed from google colab to local environment
2. target prediction values -> depolarizing error = 0, over rotation = 0.1 
3. probability distribtuion added as neural network input (two branches of input together with gate sequence input)
4. learning rate affects convergence and training greatly, is set to 1e-5
5. current constant lr decay based on steps is not optimal, need to change to a more advanced lr scheduler later based on losses trend 
6. adjustment might have to make to modify data pipeline (e.g. whether to change from NN predicting values for each data point to each mini-batch instead)
