import pandas as pd

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