# RNN---LSTM
RNN - LSTM Network train and prediction using MNIST dataset
# RNN model: the long short-term memory (LSTM) model.
- For example, in a time series where the current stock price is decided by the historical stock price, where the dependency can be short or long.
# NOTES:--
1) Once training is done, the network and network parameters are saved as pickle format. 
2) The below step in the code used for reloading the .pkl file and it is very much helpful while large data training.
          # # LOAD
          # rnn.load_state_dict(torch.load('rnn_lstm_param.pkl'))
3) # # TEST -> this step used for testing purpose.

THANKS & REGARDS,
USSAIN RASWI
