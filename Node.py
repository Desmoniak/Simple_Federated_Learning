# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

class Node():
    def __init__(self, _model, _data_train, _data_valid, _data_test):
        self.model = _model
        self.data_train = _data_train
        self.data_valid = _data_valid
        self.data_test = _data_test
    
    def train():
        pass
    
    def valid():
        pass
    
    def test():
        pass