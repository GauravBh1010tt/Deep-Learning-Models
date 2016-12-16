# Deep-Learning-Models
Implementing from scratch: NN, RNN, LSTM, GRU and CNN using theano. 

Usage:

```python
model = rnn(vocab_size = 2000,h_dim = 100 ,saved_model = False)
```
vocab_size = size of vocabulary
If using a previously saved model, use `saved_model=True`

The format the data should be 
```python
data = ['crystals in urine results',
        'picture of state trooper motorcycles',
        'chester a arthur',
        'missouri dept of elementary and secondary',
        'business forms for year end statement']
```
Prepare the data using
```python
X_train,Y_train = prepare_data(data)
```
Train the model
```python
train_with_sgd(model,X_train[:100],Y_train[:100],nepoch=3,learning_rate=0.01)
```
For LSTM and GRU
```python
model = lstm(vocab_size = 2000,h_dim = 100)
model = gru(vocab_size = 2000,h_dim = 100)
```
