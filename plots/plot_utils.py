import os
import re
import numpy as np
#import sys 
#sys.path.append("/home/moon/data/exps/ordering/")

from sim.utils.record_utils import read_fromcsv
from types import SimpleNamespace
import itertools
from datetime import datetime

TEST = 0
YLABELS = ['Round', 'Training Loss', 'Training Top1 Accuracy (%)', 'Train Top5 Accuracy (%)', 'Test Loss',  'Test Top1 Accuracy (%)', 'Test Top5 Accuracy (%)']

MARKERS = ['o', "s", 'd', '<', '>', 'v', '^', '.', "*", "+", "x", "X", 'p', 'P','D', ]

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C9']

def moving_average(data, window=5):
    r'''The length of results is: len(data)-len(windows)+1'''
    weights = np.repeat(1.0, window)/window
    return np.convolve(data, weights, 'valid')

def save_fig_timestamp(fig, format='.png', path='../figs/'):
    curr_time = datetime.now()
    filename = curr_time.strftime('%Y-%m-%d_%H-%M-%S')
    print('{}{}'.format(filename, format))
    fig.savefig('{}/{}{}'.format(path, filename, format), bbox_inches='tight', dpi=300)

def get_existing_files(files, path):
    existing_files = []
    for file in files:
        fullpath = os.path.join(path, file)
        fullpath = fullpath + '.csv'
        if os.path.exists(fullpath):
            existing_files.append(file)
    return existing_files

def grid_search(pattern, args, setup, path=0):
    # prepare files
    files = []
    combinations = itertools.product(*tuple(args.values()))
    for i in combinations:
        file = pattern.format(*i)
        files.append(file)
        #print(file)
    
    files = get_existing_files(files, path)

    # find the best settings
    best_value = 0; best_file = ''
    for file in files:
        df = read_fromcsv(file, path)
        df = df[df['round'] <= setup['end']]
        curr_value = df.iloc[:, setup['metric']].values[-setup['select']:].mean(axis=0)
        # filter some files
        if setup['filter_func'](file, curr_value):
            continue
        if setup['cmp_func'](curr_value, best_value):
            best_file = file
            best_value = curr_value
    print('{}'.format(best_file))
    return best_file

def filexcommand(args, input_pattern, output_pattern, inputs):
    outputs = []
    
    combinations = itertools.product(*tuple(args.values()))
    for i in combinations:
        input = input_pattern.formart(*i)
        if input in inputs:
            output = output_pattern.format(*i)
            outputs.append(output)
    return outputs


if __name__ == '__main__':
    pass