from utils import *
from layer import *

df = pd.read_csv('./data/data.csv')
columns_all=df.columns
#提取待使用的数据
## ph
dataset=df[["监测时间","pH"]]
## 溶解氧
dataset=df[['监测时间','溶解氧']]
## 高锰酸钾
dataset=df[['监测时间','高锰酸钾']]

dataset.columns = ["ds","y"]
dataset["ds"] = pd.to_datetime(dataset["ds"])
data = dataset["y"]
data = data[0:11347]
N = len(data)
print(N)
# 初始化结果列表
predictions = []

#格式转换
#数据归一化
scaler = MinMaxScaler()
data = np.array(data).reshape(-1, 1)
scaler.fit(data)
data = scaler.transform(data)

#VMD经验分解
data_train, data_val = train_test_split(data, test_size=0.2, random_state=42)
u, u_hat, omega = VMD(data_train, alpha, tau, K, DC, init, tol)
data1=u[1, :]
u, u_hat, omega = VMD(data_val, alpha, tau, K, DC, init, tol)
data2=u[1, :]

# 选择最优参数
best_variance = float('inf')  # 初始化最优方差为正无穷大
best_num_sifts = None
best_num_ensembles = None

for num_sifts in range(5, 10):  # 假设尝试从5到20个sift
    for num_ensembles in range(1,5):  # 假设尝试从5到20个ensemble
        variance = evaluate_variance(data1, data2, num_sifts, num_ensembles)
        if variance < best_variance:
            best_variance = variance
            best_num_sifts = num_sifts
            best_num_ensembles = num_ensembles

print("Best number of sifts:", best_num_sifts)
print("Best number of ensembles:", best_num_ensembles)

#PH，高锰酸钾：6  溶解氧：5
num_sifts=6
num_ensembles=1

epochs = pd.DataFrame([300, 300, 300, 300, 300, 300])#300，2,50，,200
model = tcb_bilstm_att(df, 6, epochs)

# 参数设置
i=0
t=3000
#method1
seq_len = 6  # 步长，即每个样本包含的时间步
window_size = 240  # 窗口大小
future_step = 120  # 预测步长
# 定义一个列表用于保存预测值
preds = []
while len(preds) < len(data):
    start = future_step * i
    end = start + window_size + future_step
    window_data = data[start:end]

    # 进行EMD分解
    data2=uVMD(window_data)
    imfs = ceemdan(data2, 6, 1)

    # 将 array 转换为 DataFrame 的一列，并命名为 'IMF1'
    df1 = pd.DataFrame({'IMF1': imfs[0][0][0]})
    # 将所有数据乘以6
    df1['IMF2'] = imfs[0][1][0][0]
    # 添加 'IMF2' 列
    df1['IMF3'] = imfs[0][2][0][0][0]
    df1['IMF4'] = imfs[0][3][0][0][0][0]
    df1['IMF5'] = imfs[0][4][0][0][0][0][0]
    df1['IMF6'] = imfs[0][5][0][0][0][0][0][0]
    seq_len=6
    l=len(df1)
    data_train = df1[l - window_size:l]
    data_test=df1[l-future_step:l]

    X_train = np.array([data_train.values[i:i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
    y_train = np.array([data_train.values[i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])
    X_test = np.array([data_test.values[i:i + seq_len, :] for i in range(data_test.shape[0] - seq_len)])
    y_test = np.array([data_test.values[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])

    # 训练子模型
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=batch_size, shuffle=False)
    y_pred = model.predict(X_test)

    preds.extend(y_pred)
    # 更新T值
    i += 1


len(preds)
len(data)

preds = np.array(preds).reshape(-1, 1)
scaler.fit(preds)
preds = scaler.transform(preds)

data = np.array(data).reshape(-1, 1)
scaler.fit(data)
data = scaler.transform(data)

rmse = np.sqrt(mean_squared_error(preds[:len(preds)],data[:len(preds)]))

mape = mean_absolute_percentage_error(preds[:len(preds)],data[:len(preds)])

mae = mean_absolute_error(preds[:len(preds)],data[:len(preds)])

srocc, _ = spearmanr(data[:len(preds)], preds)

krocc, _ = kendalltau(data[:len(preds)], preds)


#mape过大 修正
def mape_loss_func2(pred,label):
    return np.fabs((label-pred)/np.clip(label,0.1,1)).mean()
mape_1 = mape_loss_func2(preds[:len(preds)],data[:len(preds)])

pH1=[rmse, mape, mae, mape_1]
o1=[rmse, mape, mae, mape_1]
gmsj1=[rmse, mape, mae, mape_1]

print(pH1,o1,gmsj1)

#method2
newdata=data[0:3000]
while len(newdata) < len(data):
    # 选择窗口内的数据
    future_step=120
    window_size =240
    # 进行EMD分解
    data2=uVMD(newdata[future_step*i:])
    imfs = ceemdan(data2,6,1)

    # 将 array 转换为 DataFrame 的一列，并命名为 'IMF1'
    df1 = pd.DataFrame({'IMF1': imfs[0][0][0]})
    # 将所有数据乘以6
    df1['IMF2'] = imfs[0][1][0][0]
    # 添加 'IMF2' 列
    df1['IMF3'] = imfs[0][2][0][0][0]
    df1['IMF4'] = imfs[0][3][0][0][0][0]
    df1['IMF5'] = imfs[0][4][0][0][0][0][0]
    df1['IMF6'] = imfs[0][5][0][0][0][0][0][0]
    seq_len=6
    l=len(df1)
    data_train = df1[l - window_size:l]
    data_test=df1[l-future_step:l]

    X_train = np.array([data_train.values[i:i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
    y_train = np.array([data_train.values[i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])
    X_test = np.array([data_test.values[i:i + seq_len, :] for i in range(data_test.shape[0] - seq_len)])
    y_test = np.array([data_test.values[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])

    # 训练子模型
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=batch_size, shuffle=False)
    y_pred = model.predict(X_test)

    newdata = np.concatenate((newdata, y_pred*6), axis=0)
    # 更新T值
    i += 1

len(newdata)
len(data)

newdata = np.array(newdata).reshape(-1, 1)
scaler.fit(newdata)
newdata = scaler.transform(newdata)

data = np.array(data).reshape(-1, 1)
scaler.fit(data)
data = scaler.transform(data)

rmse = np.sqrt(mean_squared_error(newdata[3000:len(data)],data[3000:len(data)]))

mape = mean_absolute_percentage_error(newdata[3000:len(data)],data[3000:len(data)])

mae = mean_absolute_error(newdata[3000:len(data)],data[3000:len(data)])

srocc, _ = spearmanr(newdata[3000:len(data)],data[3000:len(data)])

krocc, _ = kendalltau(newdata[3000:len(data)],data[3000:len(data)])

#mape过大 修正
def mape_loss_func2(pred,label):
    return np.fabs((label-pred)/np.clip(label,0.1,1)).mean()
mape_2=mape_loss_func2(newdata[3000:len(data)],data[3000:len(data)])

pH2=[rmse, mape, mae, mape_2]
o2=[rmse, mape, mae, mape_2]
gmsj2=[rmse, mape, mae, mape_2]

print(pH2,o2,gmsj2)

#method3
future_step=120
window_size =240
i=0
newy=np.empty((future_step,1))
newy_pred=np.empty((future_step,1))
while len(x) < len(data):
    future_step=120
    window_size =240
    x=data[1+future_step*i:1+future_step*i+window_size]
    y=data[1+future_step*i+window_size:1+future_step*(i+1)+window_size]
    # 进行EMD分解
    data1=uVMD(x)
    imfs = ceemdan(data1,6,1)
    data2=uVMD(y)
    imfs2 = ceemdan(data2,6,1)

    #x值的
    # 将 array 转换为 DataFrame 的一列，并命名为 'IMF1'
    df1 = pd.DataFrame({'IMF1': imfs[0][0][0]})
    # 将所有数据乘以6
    df1['IMF2'] = imfs[0][1][0][0]
    # 添加 'IMF2' 列
    df1['IMF3'] = imfs[0][2][0][0][0]
    df1['IMF4'] = imfs[0][3][0][0][0][0]
    df1['IMF5'] = imfs[0][4][0][0][0][0][0]
    df1['IMF6'] = imfs[0][5][0][0][0][0][0][0]

    #y值的
    df2 = pd.DataFrame({'IMF1': imfs2[0][0][0]})
    # 将所有数据乘以6
    df2['IMF2'] = imfs2[0][1][0][0]
    # 添加 'IMF2' 列
    df2['IMF3'] = imfs2[0][2][0][0][0]
    df2['IMF4'] = imfs2[0][3][0][0][0][0]
    df2['IMF5'] = imfs2[0][4][0][0][0][0][0]
    df2['IMF6'] = imfs2[0][5][0][0][0][0][0][0]

    seq_len=6
    X_train = np.array([df1.values[i:i + seq_len, :] for i in range(df1.shape[0] - seq_len)])
    y_train = np.array([df1.values[i + seq_len, 0] for i in range(df1.shape[0] - seq_len)])
    X_test = np.array([df2.values[i:i + seq_len, :] for i in range(df2.shape[0] - seq_len)])
    y_test = np.array([df2.values[i + seq_len, 0] for i in range(df2.shape[0] - seq_len)])

    #训练
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=batch_size, shuffle=False)
    #得到y值
    y_pred = model.predict(X_test)
    newy = np.concatenate((newy, y), axis=0)
    newy_pred = np.concatenate((newy_pred, y_pred*6), axis=0)

    # 更新T值
    i += 1


len(newy_pred)

newy_pred = np.array(newy_pred).reshape(-1, 1)
scaler.fit(newy_pred)
newy_pred = scaler.transform(newy_pred)

data = np.array(data).reshape(-1, 1)
scaler.fit(data)
data = scaler.transform(data)

rmse = np.sqrt(mean_squared_error(newy_pred[3000:len(newy_pred)],data[3000:len(newy_pred)]))

mape = mean_absolute_percentage_error(newy_pred[3000:len(newy_pred)],data[3000:len(newy_pred)])

mae = mean_absolute_error(newy_pred[3000:len(newy_pred)],data[3000:len(newy_pred)])

srocc, _ = spearmanr(newy_pred[3000:len(newy_pred)],data[3000:len(newy_pred)])

krocc, _ = kendalltau(newy_pred[3000:len(newy_pred)],data[3000:len(newy_pred)])

#mape过大 修正
def mape_loss_func2(pred,label):
    return np.fabs((label-pred)/np.clip(label,0.1,1)).mean()
mape_3 = mape_loss_func2(newy_pred[3000:len(newy_pred)],data[3000:len(newy_pred)])

pH3=[rmse, mape, mae, mape_3]
o3=[rmse, mape, mae, mape_3]
gmsj3=[rmse, mape, mae, mape_3]

print(pH3,o3,gmsj3)