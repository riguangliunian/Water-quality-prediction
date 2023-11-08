from utils import *
from layer import *

epochs = pd.DataFrame([300, 300, 300, 300, 300, 300])#300，2,50，,200
t = np.array([0, 1, 3, 4, 5])
rmse_all = []
mae_all=[]
mape_all = []
all_ypred = np.zeros(0)
all_ytest = np.zeros(0)
for i in range(6):
#    df = pd.read_csv('./try.csv')
#    df = df.iloc[:, 1:]
    columns_all=df.columns
    feature = df[columns_all[i]]
    df.drop(labels=[columns_all[i]], axis=1, inplace=True)
    df.insert(0, columns_all[i], feature)
    df = df.fillna(df.interpolate())
    columns = df.columns

    time_step=6

    ytest, ypred, gamma, attention, model, X_test = tcb_bilstm_att(df, 6, int(epochs.iloc[i]))

    ypred = pd.DataFrame(ypred)
    ytest = pd.DataFrame(ytest)
    ypred.dropna(inplace=True)
    ytest.dropna(inplace=True)
    num_ypred = len(ypred)
    num_ytest = len(ytest)
    num_min = num_ypred if num_ypred < num_ytest else num_ytest
    ypred = ypred[0:num_min]
    ytest = ytest[0:num_min]

    all_ypred = np.concatenate((all_ypred, ypred.flatten()))
    all_ytest = np.concatenate((all_ytest, ytest.flatten()))

    # 计算评价指标
    rmse = np.sqrt(mean_squared_error(ytest*5, ypred*5))
    mape = mean_absolute_percentage_error(ytest*5, ypred*5)
    rmse_all.append(rmse)
    mape_all.append(mape)
    mae_all.append(mae)
    print(time_step)
    print(rmse_all)
    print(mape_all)
    print(mae_all)
