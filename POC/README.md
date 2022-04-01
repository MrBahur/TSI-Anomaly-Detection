# Taboola-Starship-Internship
Description

### Supervised model (labeled data):
We believe that the action conversion rate is correlated with the traffic and SLA in a certain data center. Thus, we would like to predict the action conversion rate based on the data center traffic rate and response time. 
You are going to build a supervised learning LSTM model that will get as input several metrics in a CSV format, such as p95, 5min rate and will try to predict the total success and failed action conversion rate. 

example for parameters:

* -path "data\Kobi_Bryant_26_1-29_1\AM" 
* -prediction "total_success_action_conversions" 
* -test_size 0.1 
* -predict_amount 6
* -ignore total_failed_action_conversions

### Unsupervised model:
We would like to have a model that can predict a data center status based on several metrics given as input.
You are going to build an unsupervised LSTM autoencoder model that gets several metrics as CSV files, and predicts their behaviour in the future. Then, you will add weights to the predicted metrics, and calculate a data center score for a future period of time.

no parameters required.
