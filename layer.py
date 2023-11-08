from utils import *
#双向多尺度TCN
def reverse_layer(x):
    return reverse(x, axes=1)

def ResBlock(x, filters, kernel_size, dilation_rate):
    # Dilated causal convolution (left-to-right)
    r1 = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
    r1 = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r1)

    # Reverse the input and perform dilated causal convolution (right-to-left)
    reversed_x = Lambda(reverse_layer)(x)
    r2 = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(reversed_x)
    r2 = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r2)

    # Reverse the result from the right-to-left convolution
    r2 = Lambda(reverse_layer)(r2)

    # Combine the results from the two directions
    r = Add()([r1, r2])

    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size=1)(x)

    o = Add()([r, shortcut])
    o = Activation('relu')(o)
    o = BatchNormalization()(o)
    o = Dropout(0.2)(o)

    return o

def TCN(x,TIME_STEPS,INPUT_DIM):
    # Multiple ResBlocks with different dilation rates
    x = Reshape((TIME_STEPS, INPUT_DIM))(x)
    x1 = ResBlock(x, filters=32, kernel_size=3, dilation_rate=1)
    x2 = ResBlock(x1, filters=32, kernel_size=3, dilation_rate=2)
    x3 = ResBlock(x2, filters=32, kernel_size=3, dilation_rate=4)
    merged = Add()([x1, x2, x3])
    merged_bidir = Conv1D(16, kernel_size=3, padding='same', activation='relu')(merged)
    return merged_bidir

def TAM_Module(x):
    gamma = tf.Variable(tf.zeros(1), name='gamma')
    x_origin = x
    proj_query, proj_key, proj_value = x, x, x
    #proj_query = Dense(proj_query.shape[2])(proj_query)
    #proj_key = Dense(proj_key.shape[2])(proj_key)
    #proj_value = Dense(proj_value.shape[2])(proj_value)
    proj_key = tf.transpose(proj_key, perm=[0, 2, 1]) # 对k进行转置,q和v不动
    energy = tf.matmul(proj_query, proj_key)
    attention = Activation('softmax')(energy)
    #attention = tf.nn.softmax(energy, name='attention')
    proj_value = tf.transpose(proj_value, perm=[0, 2, 1]) #对v进行转置
    out = tf.matmul(proj_value, attention)
    out = tf.transpose(out, perm=[0, 2, 1])
    out = add([out*gamma, x_origin])
    out = BatchNormalization()(out)
    return out, gamma, attention

time_step=6
t1 = 2

def tcb_bilstm_att(df, time_step, epochs):
#接下来的code组要是构建一个模型方便运行
    adam = Adam(learning_rate=0.005)
    TN = initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=123)
#    data_train = df.iloc[:int(df.shape[0] * 0.8), :]
#    data_test = df.iloc[int(df.shape[0] * 0.8):, :]
#    max = np.max(df1.iloc[:, 0])
#    min = np.min(df1.iloc[:, 0])
#    数据归一化
#    scaler = MinMaxScaler()
#    scaler.fit(data_train)
#    data_train = scaler.transform(data_train)
#    data_test = scaler.transform(data_test)

    output_dim = 1
    # 每轮训练模型时，样本的数量
    batch_size = 512
    # 训练60轮次
    #epochs = 400
    seq_len = time_step
    hidden_size = 128
    TIME_STEPS = 6
    INPUT_DIM = 6
    lstm_units = 4

    # 数据格式处理
    X_train = np.array([df1.values[i:i + seq_len, :] for i in range(df1.shape[0] - seq_len)])
    y_train = np.array([df1.values[i + seq_len, 0] for i in range(df1.shape[0] - seq_len)])
    X_test = np.array([df2.values[i:i + seq_len, :] for i in range(df2.shape[0] - seq_len)])
    y_test = np.array([df2.values[i + seq_len, 0] for i in range(df2.shape[0] - seq_len)])

    #EarlyStop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min', min_delta=0.0001)

    inputs = Input(shape=(TIME_STEPS, INPUT_DIM))

    # 进行TCN
    x = TCN(inputs,TIME_STEPS, INPUT_DIM)
    # x=TCN(inputs)
    new_time_steps = x.shape[1]

    # 进行单向LSTM
    lstm_out = Bidirectional(LSTM(lstm_units, activation='sigmoid', return_sequences=True, kernel_initializer=TN))(x)
    #lstm_out = LSTM(lstm_units, activation='sigmoid', return_sequences=True, kernel_initializer=TN)(x)
    tam, gamma, attention = TAM_Module(lstm_out)
    # lstm_out = LSTM(lstm_units, activation='tanh', return_sequences=False, kernel_initializer=TN)(lstm_out)

    # 做FC并构建模型
    output = Flatten()(x)
    output = Dense(1, kernel_initializer=TN)(output)
    model = Model(inputs=inputs, outputs=output)

    # 模型编译
    model.compile(loss='mean_squared_error', optimizer=adam)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, shuffle=False)
    y_pred = model.predict(X_test)
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('Model loss')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()
    ytest = (max - min) * y_test + min
    ypred = (max - min) * y_pred + min
    return ytest, ypred, gamma, attention, model, X_test