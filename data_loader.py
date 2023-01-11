from sklearn.model_selection import train_test_split
from Utils.DataUtils import DataUtils
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt





class TrainUtils:
    def __init__(self) -> None:
        super().__init__()
        self.dataUtils = DataUtils()

    def prepareData(self, delay=5):
        ecgAll, fecg, fqrs_rpeaks = self.dataUtils.readData(1)
        ecgAll = ecgAll[range(1), :]
        delayNum = ecgAll.shape[0]
        fecgAll = self.dataUtils.createDelayRepetition(fecg, delayNum, delay)
        for i in range(2, 5):
            ecg, fecg,fqrs_rpeaks1 = self.dataUtils.readData(i)
            ecg = ecg[range(1), :]
            
            fqrs_rpeaks1 = fqrs_rpeaks1 + 60000*(i-1)
            fqrs_rpeaks= np.append(fqrs_rpeaks,fqrs_rpeaks1)
            fecgDelayed = self.dataUtils.createDelayRepetition(fecg, delayNum, delay)
            ecgAll = np.append(ecgAll, ecg, axis=1)
            fecgAll = np.append(fecgAll, fecgDelayed, axis=1)

        ecgWindows, fecgWindows, fqrs_rpeaks = self.dataUtils.windowingSig(ecgAll, fecgAll, fqrs_rpeaks, windowSize=128)
        # fecgWindows = self.dataUtils.adaptFilterOnSig(ecgWindows, fecgWindows)
        # ecgWindows = self.dataUtils.calculateICA(ecgWindows, component=2)
        return ecgWindows, fecgWindows, fqrs_rpeaks

    def trainTestSplit(self, sig, label, trainPercent, shuffle=False):
        X_train, X_test, y_train, y_test = train_test_split(sig, label, train_size=trainPercent, shuffle=shuffle)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        return X_train, X_test, y_train, y_test




class Data_Loader():
    def __init__(self, FECG=True):
        super().__init__()
        self.trainUtils = TrainUtils()

    def test_trainSignal(self):
        ecgWindows, fecgWindows,fqrs_rpeaks = self.trainUtils.prepareData(delay=5)
        X_train, X_test, Y_train, Y_test = self.trainUtils.trainTestSplit(ecgWindows, fecgWindows, len(ecgWindows)-1)

        X_train = np.reshape(X_train, [-1, X_train.shape[2], X_train.shape[1]])
        X_test = np.reshape(X_test, [-1, X_test.shape[2], X_test.shape[1]])
        Y_train = np.reshape(Y_train, [-1, Y_train.shape[2], Y_train.shape[1]])
        Y_test = np.reshape(Y_test, [-1, Y_test.shape[2], Y_test.shape[1]])
        # print(X_train.shape)
        
        
        return X_train, X_test, Y_train, Y_test, fqrs_rpeaks

class Data_Item():
    def __init__(self, FECG=True):
        super().__init__()
        self.X_train, self.X_test, self.Y_train, self.Y_test, self.fqrs_rpeaks = Data_Loader().test_trainSignal()

    
    
class FECGDataset(Dataset):
    def __init__(self, data_item, train=True):
        super(FECGDataset, self).__init__()
        self.train = train
        self.X_train, self.X_test, self.Y_train, self.Y_test,self.fqrs_rpeaks = data_item.X_train, data_item.X_test, data_item.Y_train, data_item.Y_test,data_item.fqrs_rpeaks
        self.numer = len(self.X_train)
        
    def __getitem__(self, index):   
        
        if self.train:
            # print(self.X_train[index,:,:].shape)
            xx = self.X_train[index,:,:]
            yy = self.Y_train[index,:,:]
            fqrs = self.fqrs_rpeaks[index]
            
            
            y_max_index = np.argmax(yy,axis=-1)
            y_min_index = np.argmin(yy,axis=-1)
            
            x_max = xx[0,y_max_index]
            x_min = xx[0,y_min_index]
            
            t_max = 10000
            t_min = 0
            for coo in fqrs:
                v_max = np.max(xx[0,max(coo-10,0):min(coo+10,127)])
                if t_max > v_max:
                    t_max = v_max
                    v_min = np.min(xx[0,max(coo-10,0):min(coo+10,127)])
                    t_min = v_min
            
                

            if t_max == np.max(xx[0,:]) or t_min == np.min(xx[0,:]):
                index1 = index-1 if (index+1 == self.numer) else index+1
                xx = self.X_train[index1,:,:]
                yy = self.Y_train[index1,:,:]
                fqrs = self.fqrs_rpeaks[index1]
                
                
                y_max_index = np.argmax(yy,axis=-1)
                y_min_index = np.argmin(yy,axis=-1)
                
                x_max = xx[0,y_max_index]
                x_min = xx[0,y_min_index]
                
                t_max = 10000
                t_min = 0
                for coo in fqrs:
                    v_max = np.max(xx[0,max(coo-10,0):min(coo+10,127)])
                    if t_max > v_max:
                        t_max = v_max
                        t_min = np.min(xx[0,max(coo-10,0):min(coo+10,127)])
                    
            
            # plt.plot(yy.transpose(),'b')
            # plt.show()
            
            # plt.plot(xx.transpose(),'r')
            # plt.show()
            min_max_scaler = MinMaxScaler(feature_range=[t_min, t_max], copy=False)           
            yy_minmax = min_max_scaler.fit_transform(yy.transpose())
            MECG_signal =  xx - yy_minmax.transpose()
            plt.plot(MECG_signal.transpose(),'r')
            plt.title('MECG')
            plt.show()
            
            M_index = np.argmax(MECG_signal,axis=-1)
            noise = MECG_signal.copy()
            noise[0,int(max(0,M_index-15)):int(min(MECG_signal.shape[-1],M_index+15))] = noise[0,max(0,M_index-15)]
            
            # plt.plot(noise.transpose(),'r')
            # plt.title('noise')
            # plt.show()
            
            
            min_max_scaler = MinMaxScaler(feature_range=[-1, 1], copy=False)           
            MECG_signal = min_max_scaler.fit_transform(MECG_signal.transpose()).transpose()

            AECG_signal = min_max_scaler.fit_transform(self.X_train[index,:,:].transpose()).transpose()
            FECG_signal = min_max_scaler.fit_transform(self.Y_train[index,:,:].transpose()).transpose()
            BIAS_signal = min_max_scaler.fit_transform(noise.transpose()).transpose()
            
            
            return AECG_signal,FECG_signal,MECG_signal,BIAS_signal
        else:
            xx = self.X_test[index,:,:]
            yy = self.Y_test[index,:,:]
            fqrs = self.fqrs_rpeaks[index]
            
            
            y_max_index = np.argmax(yy,axis=-1)
            y_min_index = np.argmin(yy,axis=-1)
            
            x_max = xx[0,y_max_index]
            x_min = xx[0,y_min_index]
            
            t_max = 10000
            t_min = 0
            for coo in fqrs:
                v_max = np.max(xx[0,max(coo-10,0):min(coo+10,127)])
                if t_max > v_max:
                    t_max = v_max
                    v_min = np.min(xx[0,max(coo-10,0):min(coo+10,127)])
                    t_min = v_min
            
                

            if t_max == np.max(xx[0,:]) or t_min == np.min(xx[0,:]):
                index1 = index-1 if (index+1 == self.numer) else index+1
                xx = self.X_test[index1,:,:]
                yy = self.Y_test[index1,:,:]
                fqrs = self.fqrs_rpeaks[index1]
                
                
                y_max_index = np.argmax(yy,axis=-1)
                y_min_index = np.argmin(yy,axis=-1)
                
                x_max = xx[0,y_max_index]
                x_min = xx[0,y_min_index]
                
                t_max = 10000
                t_min = 0
                for coo in fqrs:
                    v_max = np.max(xx[0,max(coo-10,0):min(coo+10,127)])
                    if t_max > v_max:
                        t_max = v_max
                        t_min = np.min(xx[0,max(coo-10,0):min(coo+10,127)])
                    
            
            min_max_scaler = MinMaxScaler(feature_range=[t_min, t_max], copy=False)           
            yy_minmax = min_max_scaler.fit_transform(yy.transpose())
            MECG_signal =  xx - yy_minmax.transpose()
            
            M_index = np.argmax(MECG_signal,axis=-1)
            noise = MECG_signal.copy()
            noise[0,int(max(0,M_index-15)):int(min(MECG_signal.shape[-1],M_index+15))] = noise[0,max(0,M_index-15)]
            
            return self.X_test[index,:,:],self.Y_test[index,:,:],MECG_signal,noise
        
    def __len__(self):
        return len(self.X_train)