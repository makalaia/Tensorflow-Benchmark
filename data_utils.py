import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from scipy.fftpack import fft, ifft, fftshift
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy.signal import correlate


class BestHyperparameters(dict):
    """
    Classe que é basicamente um dicionário de dados, com suporte a pickle. Usada para gravar algumas informacoes quanto
    aos melhores hyperparametros da otimização.
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


def create_dataset(dataset, look_back=1, nDerivative=0):
    """
    Funcao que, a partir de um dataset, cria outro com uma janela de lag e de derivativo
    @param dataset: dataset a ser modificado
    @param look_back: tamanho da janela de lag
    @param nDerivative: tamanho da janela de derivativo
    @return: novo dataset
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), -1]
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])
    k = dataset[look_back:, :-1]
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    dataX = np.concatenate((k, dataX), axis=1)
    k = addDerivativeTerms(dataset, nDerivative)
    k = k[look_back - nDerivative - 1:, -nDerivative:]
    dataX = np.concatenate((dataX, k), axis=1)
    data = np.hstack((np.array(dataX), np.reshape(dataY, (dataY.shape[0], 1))))
    return data


def calculateMse(real, predict):
    """
    Funcao que calcula o MSE a partir dos dados reais e preditos.
    @param real: lista com os valores reais
    @param predict: lista com os valores preditos
    @return: valor do MSE calculado
    """
    return ((real - predict) ** 2).mean(axis=0)


def calculate_rmspe(real, predict):
    """
    Funcao que calcula o RMSPE a partir dos dados reais e preditos.
    @param real: lista com os valores reais
    @param predict: lista com os valores preditos
    @return: valor do RMSPE calculado
    """
    if isinstance(real, pd.DataFrame):
        real = real.values
    if isinstance(predict, pd.DataFrame):
        predict = predict.values
    real = real.flatten();
    predict = predict.flatten();
    if (real.shape != predict.shape):
        raise Exception('The shapes must be the same to perform the calculation Real: %d\tPredict: %d' %
                        (real.shape, predict.shape))
    m = len(real)
    t = np.divide((real - predict), real, out=np.zeros_like(real), where=real != 0)
    return np.sqrt(np.sum(np.power(t, 2)) / m)


def calculate_rmse(real, predict):
    """
    Funcao que calcula o RMSEE a partir dos dados reais e preditos.
    @param real: lista com os valores reais
    @param predict: lista com os valores preditos
    @return: valor do RMSE calculado
    """
    if isinstance(real, pd.DataFrame):
        real = real.values
    if isinstance(predict, pd.DataFrame):
        predict = predict.values
    real = real.flatten();
    predict = predict.flatten();
    if real.shape != predict.shape:
        raise Exception('The shapes must be the same to perform the calculation Real: %s\tPredict: %s' %
                        (real.shape, predict.shape))
    m = len(real)
    return np.sqrt(np.sum(np.power((real - predict), 2)) / m)


def addDerivativeTerms(data, n):
    """
    Metodo que adiciona termos derivativos ao dataset
    @param data: dataset
    @param n: numero de termos derivativos a serem adicionados
    @return: dataset com os termos adicionados
    """
    if n == 0:
        return data
    x = data[n + 1:, :-1]
    y = data[:, -1]
    for i in range(0, n):
        # t = y[i:] - data[i:i-n, -1]
        t = y[i + 1:-n + i] - y[i:-n + i - 1]
        t = t[:, None]
        x = np.concatenate((x, t), axis=1)
    return x


def addFourier(dataset):
    """
    Metodo que aplica transformada rapida de fourier a ultima coluna e a adiciona como dado
    @param dataset: dataset em que se deseja adicionar a variável
    @return: dataset modificado
    """
    y = dataset[:, -1]
    k = abs(fft(y, len(y) * 2))
    k = k[:int(len(k) / 2)]
    k = np.reshape(k, (len(k), 1))
    data = np.concatenate((k, dataset), axis=1)
    return data


def addLog(dataset):
    """
    Metodo que aplica log10 em todas as colunas de parâmetro do dataset, e as adiciona como parametros.
    Indicado para distribuições Gaussianas.
    @param dataset: dataset em que deseja alterar
    @return: dataset modificado
    """
    y = dataset[:-1, -1:]
    y = np.log10(y)
    datalog = np.concatenate((y, dataset[1:, :]), axis=1)
    return datalog


def get_max_error(real, predict):
    """
    Metodo que procura qual o maior erro foi cometido, comparando os dois vetores valor por valor.
    @param real: vetor com as informações base
    @param predict: vetor em que se deseja comparar com o real
    @return: maximo erro em porcentagem
    """
    if isinstance(real, pd.DataFrame):
        real = real.values
    if isinstance(predict, pd.DataFrame):
        predict = predict.values
    erro_max = 0
    for r, p in zip(real, predict):
        if r != 0:
            dif = np.abs((r - p) / r)
        else:
            return 0
        if dif > erro_max:
            erro_max = dif
    return erro_max[0] * 100


def get_avg_error(real, predict):
    """
    Metodo que procura qual o erro total que foi cometido, dado pela soma dos dois vetores, dividido pela soma do real.
    @param real: vetor com as informações base
    @param predict: vetor em que se deseja comparar com o real
    @return: erro total em porcentagem
    """
    if isinstance(real, pd.DataFrame):
        real = real.values
    if isinstance(predict, pd.DataFrame):
        predict = predict.values
    r = np.sum(real)
    p = np.sum(predict)
    return (r - p) / r * 100


def cross_correlation(x, y, shift=1):
    y = y[shift:]
    x = x[:-shift]
    lag = np.argmax(correlate(x, y))
    r = np.roll(x, shift=int(np.ceil(lag)))
    return r


def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)


# shift < 0 means that y starts 'shift' time steps before x shift > 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift


def get_errors(real, predict):
    errors = dict()
    errors['rmse'] = calculate_rmse(real, predict)
    errors['rmspe'] = calculate_rmspe(real, predict) * 100
    errors['max_error'] = get_max_error(real, predict)
    errors['avg_error'] = get_avg_error(real, predict)
    return errors


def printErrors(real, predict):
    errors = get_errors(real, predict)
    print('RMSE: %.2f' % errors['rmse'])
    print('RMSPE: %.2f%%' % errors['rmspe'])
    print('AVG ERROR: %.2f%%' % errors['avg_error'])
    print('MAX ERROR: %.2f%%' % errors['max_error'])


def plot(y, y_trained, y_val, y_test=None, tittle='Previsao', margin=None):
    if isinstance(y, pd.DataFrame):
        y = y.values
    if isinstance(y, pd.DataFrame):
        y_trained = y_trained.values
    if isinstance(y, pd.DataFrame):
        y_val = y_val.values
    if isinstance(y, pd.DataFrame):
        y_val = y_val.values

    train_size = len(y_trained)
    val_size = len(y_val)
    y_train = y[:train_size]
    # print('Margem de Erro: ' + str(margin))

    # plot
    y_val = np.ravel(y_val)
    margem = None
    if margin is None:
        margem = calculate_rmse(y_train, y_trained)
    else:
        margem = margin * y_val
    plt.plot(y, linewidth=2, label='Real Data', color='blue')
    plt.plot(y_trained, label='Train Data', color='green')
    x_val = np.arange(train_size - 1, train_size + val_size - 1)
    plt.plot(x_val, y_val, label='Validation Data', color='red')
    plt.fill_between(x_val, y_val - margem, y_val + margem, facecolor='red', alpha=.5)
    if y_test is not None:
        if margin is None:
            margem = calculate_rmse(y_train, y_trained)
        else:
            y_test = np.ravel(y_test)
            y_test = np.hstack((y_val[-1:], y_test))
            margem = np.ravel(margin * y_test)
        test_size = len(y_test)
        x_test = np.arange(train_size + val_size - 2, train_size + val_size - 2 + test_size)
        plt.plot(x_test, y_test, label='Test Data', color='black')
        plt.fill_between(x_test, y_test - margem, y_test + margem, facecolor='black', alpha=.5)
    plt.title(tittle)
    plt.show()


def __plot_bar__(real, predict, index=None, title=None, confidence=None, industria=None):
    if real.shape != predict.shape:
        raise ValueError('All input dimensions must agree.')
    plt.subplots()
    ind = np.arange(0, real.size)
    bar_width = .35
    if industria is not None and industria.size == real.size:
        bar_width = .25
        plt.bar(ind +bar_width*2, industria, bar_width,
                alpha=.5,
                color='g',
                label='predict-ST')

    if index is not None:
        plt.xticks()
        plt.xticks(ind + bar_width, index)
    if title is not None:
        plt.title(title)
    plt.bar(ind, real, bar_width,
            alpha=.5,
            color='b',
            label='real')
    if confidence is not None:
        plt.bar(ind + bar_width, predict, bar_width,
                alpha=.5,
                color='r',
                yerr=confidence,
                error_kw={'ecolor': '0.3'},
                label='predict-IA')
    else:
        plt.bar(ind + bar_width, predict, bar_width,
                alpha=.5,
                color='r',
                label='predict-IA')
    plt.legend()
    plt.show()


def plot_bar(y_total, y_pred, n_meses, index=None, title=None, confidence=None, val_size=150, industria=None):
    if not isinstance(y_total, pd.DataFrame):
        raise TypeError('y must be a Dataframe to continue')
    real = np.zeros(n_meses)
    pred = np.zeros(n_meses)
    ind = list()
    for i in range(1, n_meses + 1):
        if i < 10:
            ind.append('01/0' + str(i) + '/2017')
        else:
            ind.append('01/' + str(i) + '/2017')
        index_pos = np.where(y_total.index == ind[i - 1])[0][0]
        real[i - 1] = y_total.iloc[index_pos]
        pred[i - 1] = y_pred[index_pos]

    # confidence
    if confidence is not None:
        if confidence > 1:
            raise ValueError('Confidence must be a value between 0 and 1')
        yerr = np.zeros(n_meses)
        confidence += (1-confidence)/2
        confidence = st.norm.ppf(confidence)
        for i in range(1, n_meses + 1):
            index_pos = np.where(y_total.index == ind[i - 1])[0][0]
            yerr[i - 1] = confidence*calculate_rmse(y_total.iloc[index_pos-val_size:index_pos], y_pred[index_pos-val_size:index_pos])

    __plot_bar__(real, pred, index=index, title=title, confidence=yerr, industria=industria)


def remove_outliers(df, quantile_max=.995):
    out_mask = df > df.quantile(quantile_max)
    df = df.mask(out_mask, df.quantile(quantile_max), axis=1)
    return df


def reduce_dimensions(x, dimensions):
    scaler = StandardScaler();
    if isinstance(x, np.ndarray):
        x = scaler.fit_transform(x);
        pca = PCA(n_components=dimensions)
        x = pca.fit_transform(x)
        return x, pca, scaler
    elif isinstance(x, pd.DataFrame):
        x = scaler.fit_transform(x);
        pca = PCA(n_components=dimensions)
        pca.fit(x)
        x = pd.DataFrame(pca.fit_transform(x))
        return x, pca, scaler

def shuffle_data(x, y):
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return x[p], y[p]