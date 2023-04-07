import pandas as pd
import torch

'''
@Param value: a string value with this format "Hour(s):Minute(s):Second(s)" 
'''
def convert_to_seconds(value):
        if not any(char.isdigit() for char in value):
            return pd.NaT
        hour, minute, second = map(int, value.split(':'))
        if hour >= 15:
            hour = 0
        if minute >= 60:
            minute = 0
            hour += 1
        if second >= 60:
            second = 0
            minute += 1
        return hour*3600+minute*60+second
    
def train_valid_test(dataset, train_size=0.6, valid_size=0.2, test_size=0.2):
    return torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    