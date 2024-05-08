import pandas as pd
import warnings 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def predict_mobile_price_range(user_input):
    warnings.filterwarnings('ignore')
    df = pd.read_csv('train.csv')
    x = df.drop(columns=['price_range','m_dep','fc','talk_time'],axis = 1)    
    y = df['price_range']

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)

    sv = SVC(kernel='linear', random_state=23)
    sv.fit(x_train, y_train)

    y_pred = sv.predict(user_input)

    if y_pred == 0:
        return "50 - 300 $"
    elif y_pred == 1:
        return "300 - 600 $"
    elif y_pred == 2:
        return "600 - 900 $"
    else:
        return "900 - 1300 $"