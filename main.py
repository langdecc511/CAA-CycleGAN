
from parameter import *
from trainer import Trainer
# from tester import Tester
from data_loader import Data_Loader,Data_Item
from torch.backends import cudnn
from utils import make_folder


from torch.utils.data import DataLoader
from data_loader import FECGDataset


def main(config):
    
    data_item = Data_Item()
    train_dataset = FECGDataset(data_item,train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)   
    val_dataset = FECGDataset(data_item,train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)    
    test_dataset = FECGDataset(data_item,train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=0)
    
    
    
    # For fast training
    cudnn.benchmark = True
    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        if config.model=='sagan':
            trainer = Trainer(train_dataloader, config)
        elif config.model == 'qgan':
            trainer = qgan_trainer(train_dataloader, config)
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), config)
        tester.test()

if __name__ == '__main__':
    config = get_parameters()
    # print(config)
    main(config)