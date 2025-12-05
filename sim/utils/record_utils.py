import pandas as pd
import logging
import os

LogLevel = {
    'debug': logging.DEBUG, # 10
    'info' : logging.INFO, # 20
}

def prcolor(info, color='white'):
    # ANSI code 31 represents standard intensity red, while ANSI code 91 represents bright intensity red.
    ansi_code_color = {'red': 91, 'green': 92, 'yellow': 93, 'blue': 94, 'magenta': 95}
    if color in ansi_code_color.keys():
        print("\033[{}m{}\033[00m" .format(ansi_code_color[color], info))
    else:
        print(info)

def logconfig(path='./save/', name:str='', level='info', mode='print'):
    level = LogLevel[level]
    if 'log' in mode:
        filepath = path+'{}.log'.format(name)
        if (os.path.exists(filepath)):
            os.remove(filepath)
        logging.basicConfig(filename=filepath, level=level)
    else:
        pass

def add_log_info(info:str='', level='info', mode:str='print', color:str='white'):
    level = LogLevel[level]
    if 'print' in mode:
        if logging.INFO >= level:
            prcolor(info, color)
        else:
            pass
    if 'log' in mode:
        logging.info(info)

def add_log_debug(info:str='', level='info', mode:str='print', color:str='white'):
    level = LogLevel[level]
    if 'print' in mode:
        if logging.DEBUG >= level:
            prcolor(info, color)
        else:
            pass
    if 'log' in mode:
        logging.debug(info)


def record_exp_result(filename, result):
    savepath = './save/'
    filepath = '{}/{}.csv'.format(savepath, filename)
    round, result = result[0], result[1:]
    if round == 0:
        if (os.path.exists(filepath)):
            os.remove(filepath)
        with open (filepath, 'a+') as f:
            f.write('{},{}\n'.format('round', ','.join(['train_loss', 'train_top1', 'train_top5', 'test_loss', 'test_top1', 'test_top5'])))
    else:
        with open (filepath, 'a+') as f:
            f.write('{},{}\n'.format(round, ','.join(['{:.4f}'.format(i) for i in result])))


def record_tocsv(name, path='./save/', **kwargs):
    #epoch = [i for i in range(1, len(kwargs[list(kwargs.keys())[0]])+1)]
    #df = pd.DataFrame(kwargs)  
    #df = pd.DataFrame.from_dict(kwargs, orient='index')   
    df = pd.DataFrame(pd.DataFrame.from_dict(kwargs, orient='index').values.T, columns=list(kwargs.keys()))
    file_name = path + name + ".csv"    
    df.to_csv(file_name, index = False)

def read_fromcsv(name, path='./save/'):
    if '.csv' in name:
        df = pd.read_csv("{}{}".format(path, name))
    else:
        df = pd.read_csv("{}{}.csv".format(path, name))
    return df

def record_toexcel(name, **kwargs):
    path = './save/'
    #epoch = [i for i in range(1, len(kwargs[list(kwargs.keys())[0]])+1)]
    df = pd.DataFrame(kwargs)     
    file_name = path + name + ".xls"    
    df.to_excel(file_name, sheet_name= "Sheet1", index = False) 

def read_fromexcel(name):
    path = './save/'
    df = pd.read_excel("{}{}.xls".format(path, name), sheet_name="Sheet1") # Sheet1
    return df

def exceltocsv():
    path = './save/'
    name='client alex cifar layer 1'
    df = pd.read_excel("{}{}.xls".format(path, name)) # Sheet1
    df.to_csv("{}{}.csv".format(path, name), index = False)

if __name__ == '__main__':
    exceltocsv()
    


