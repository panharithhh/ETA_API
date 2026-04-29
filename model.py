import pandas as pd 
from sklearn.linear_model import LinearRegression 
import numpy as np
from numpy import radians, sin, cos, sqrt, arcsin
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
import joblib

#order_id,region_id,city,courier_id,lng,lat,aoi_id,aoi_type,
#accept_time,accept_gps_time,accept_gps_lng,accept_gps_lat,
#delivery_time,delivery_gps_time,delivery_gps_lng,delivery_gps_lat,ds
model = joblib.load("model.pkl")

def train_model():
    
    df = load_data() 
    clean_df = clean_data(df) 
    accept_gps_lng= clean_df["accept_gps_lng"]
    accept_gps_lat = clean_df["accept_gps_lat"]
    delivery_gps_lng = clean_df["delivery_gps_lng"]
    delivery_gps_lat =clean_df["delivery_gps_lat"]
    
    clean_df["accept_time"]  = pd.to_datetime(clean_df["accept_time"], format="%m-%d %H:%M:%S")
    clean_df["delivery_time"] = pd.to_datetime(clean_df["delivery_time"], format="%m-%d %H:%M:%S")
    
    clean_df["time_taken"] = (clean_df["delivery_time"] - clean_df["accept_time"]).dt.total_seconds() / 60

    clean_df = clean_df[clean_df["time_taken"] <= 300]

    clean_df["distance"] = haversine(accept_gps_lat, accept_gps_lng, delivery_gps_lat, delivery_gps_lng)
    clean_df = clean_df[clean_df["distance"] <= 50]
    clean_df["speed"] = clean_df["distance"] / (clean_df["time_taken"] / 60)
    clean_df["hour"] = clean_df["accept_time"].dt.hour
    clean_df["day_of_week"] = clean_df["accept_time"].dt.day_of_week

    X = clean_df[["accept_gps_lng", "accept_gps_lat", "delivery_gps_lng", "delivery_gps_lat", "distance", "hour", "day_of_week"]]
    y = clean_df["time_taken"]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)
    
    randomForestModel = RandomForestRegressor(
        n_estimators=300,       
        max_depth=None,            
        min_samples_split=5,    
        min_samples_leaf=2,      
        max_features=0.5,     
        random_state=42,
        n_jobs=-1
    )
    
    randomForestModel.fit(X_train, y_train)
    joblib.dump(randomForestModel, "model.pkl")

    predict = randomForestModel.predict(X_test)

    test_rmse = np.sqrt(np.mean((predict - y_test.values) ** 2))
    err_rate = test_rmse / y_train.std()
    print(f"Error rate: {err_rate:.4f}")
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
     


def load_data(): 
    n = 30000  
    
    drop_cols = ["region_id", "city", "courier_id", "lng", "lat", "aoi_id", "aoi_type", "accept_gps_time", "delivery_gps_time", "ds"]
    data_cq = pd.read_csv("LaDe/delivery/delivery_cq.csv").sample(n, random_state= 42).drop(columns=drop_cols)
    data_hz = pd.read_csv("LaDe/delivery/delivery_hz.csv").sample(n, random_state= 42).drop(columns=drop_cols)
    data_jl = pd.read_csv("LaDe/delivery/delivery_jl.csv").sample(n, random_state= 42).drop(columns=drop_cols)
    data_sh = pd.read_csv("LaDe/delivery/delivery_sh.csv").sample(n, random_state= 42).drop(columns=drop_cols)
    data_yt = pd.read_csv("LaDe/delivery/delivery_yt.csv").sample(n, random_state= 42).drop(columns=drop_cols)

    total_data = pd.concat([data_sh, data_cq, data_hz, data_jl, data_yt], ignore_index= True)

    return total_data

def clean_data(df : pd.DataFrame):
    df = df.dropna()
    df = df.drop_duplicates(subset="order_id", keep="first")
    return df
     

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * arcsin(sqrt(a)) 


    
def predict(data):
    distance = haversine(
        data.accept_gps_lat, data.accept_gps_lng,
        data.delivery_gps_lat, data.delivery_gps_lng
    )
    
  
    hour = data.accept_time.hour
    day_of_week = data.accept_time.weekday()
 
    features = pd.DataFrame([{
        "accept_gps_lng": data.accept_gps_lng,
        "accept_gps_lat": data.accept_gps_lat,
        "delivery_gps_lng": data.delivery_gps_lng,
        "delivery_gps_lat": data.delivery_gps_lat,
        "distance": distance,
        "hour": hour,
        "day_of_week": day_of_week,
    }])
    
   
    predicted_minutes = float(model.predict(features)[0])
    
    return predicted_minutes

train_model()
