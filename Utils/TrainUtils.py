from sklearn.model_selection import train_test_split

from Utils.DataUtils import DataUtils,DataUtils_NIFECGDB
import numpy as np


class TrainUtils:
    def __init__(self) -> None:
        super().__init__()
        self.dataUtils = DataUtils()

    def prepareData(self, delay=5):
        ecgAll, fecg,fqrs_rpeaks = self.dataUtils.readData(0)
        ecgAll = ecgAll[range(1), :]
        delayNum = ecgAll.shape[0]
        fecgAll = self.dataUtils.createDelayRepetition(fecg, delayNum, delay)
        for i in range(1, 5):
            ecg, fecg,fqrs_rpeaks1 = self.dataUtils.readData(i)
            ecg = ecg[range(1), :]
            
            
            fqrs_rpeaks1 = fqrs_rpeaks1 + 75000*i
            fqrs_rpeaks= np.append(fqrs_rpeaks,fqrs_rpeaks1)
            fecgDelayed = self.dataUtils.createDelayRepetition(fecg, delayNum, delay)
            ecgAll = np.append(ecgAll, ecg, axis=1)
            fecgAll = np.append(fecgAll, fecgDelayed, axis=1)

        ecgWindows, fecgWindows = self.dataUtils.windowingSig(ecgAll, fecgAll, windowSize=200)
        # fecgWindows = self.dataUtils.adaptFilterOnSig(ecgWindows, fecgWindows)
        # ecgWindows = self.dataUtils.calculateICA(ecgWindows, component=2)
        return ecgWindows, fecgWindows

    def trainTestSplit(self, sig, label, trainPercent, shuffle=True):
        X_train, X_test, y_train, y_test = train_test_split(sig, label, train_size=trainPercent, shuffle=shuffle)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        return X_train, X_test, y_train, y_test
    
    
    
class TrainUtils_NIFECGDB:
    def __init__(self) -> None:
        super().__init__()
        self.dataUtils = DataUtils_NIFECGDB()

    def prepareData(self, delay=5):
        ecgAll, fecg = self.dataUtils.readData(0)
        ecgAll = ecgAll[range(2,3), :]
        delayNum = ecgAll.shape[0]
        fecgAll = self.dataUtils.createDelayRepetition(fecg, delayNum, delay)

        ecgWindows, fecgWindows = self.dataUtils.windowingSig(ecgAll, fecgAll, windowSize=200)
        # fecgWindows = self.dataUtils.adaptFilterOnSig(ecgWindows, fecgWindows)
        # ecgWindows = self.dataUtils.calculateICA(ecgWindows, component=2)
        return ecgWindows, fecgWindows

    def trainTestSplit(self, sig, label, trainPercent, shuffle=False):
        X_train, X_test, y_train, y_test = train_test_split(sig, label, train_size=trainPercent, shuffle=shuffle)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        return X_train, X_test, y_train, y_test
