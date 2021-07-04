from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, ELU, Dense
from tensorflow.keras import regularizers

class RNN():
    
    def __init__(self, X_train_shape):
        self.input_shape = (X_train_shape[1], X_train_shape[2])
        self.model = Sequential()
        self.model.add(LSTM(units=24, input_shape=self.input_shape,
                            activation='tanh', 
                            recurrent_activation='sigmoid', 
                            use_bias=True, return_sequences=True))
        self.model.add(LSTM(units=12, input_shape=self.input_shape,
                            activation='tanh', recurrent_activation='sigmoid', 
                            use_bias=True, return_sequences=False))
        self.model.add(Dense(units=8))
        self.model.add(ELU(alpha=1))
        self.model.add(Dense(units=1, activation='relu'))
        
class RNNL1():
    
    def __init__(self, X_train_shape):
        self.input_shape = (X_train_shape[1], X_train_shape[2])
        self.model = Sequential()
        self.model.add(LSTM(units=30, input_shape=self.input_shape,
                            activation='tanh', 
                            recurrent_activation='sigmoid', 
                            use_bias=True, return_sequences=True,
                            kernel_regularizer=regularizers.l1(1e-05)))
        self.model.add(LSTM(units=20, input_shape=self.input_shape,
                            activation='tanh', recurrent_activation='sigmoid', 
                            use_bias=True, return_sequences=False,
                            kernel_regularizer=regularizers.l1(1e-05)))
        self.model.add(Dense(units=10,
                            kernel_regularizer=regularizers.l1(1e-05)))
        self.model.add(ELU(alpha=1))
        self.model.add(Dense(units=1, activation='relu'))

class FFNN():
    
    def __init__(self, X_train_shape):
        
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=X_train_shape[1], 
                             activation='tanh'))
        self.model.add(Dense(12, activation='tanh'))
        self.model.add(Dense(8))
        self.model.add(Dense(1, activation='relu'))