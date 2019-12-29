
import pandas as pd
import os
from math import *





class FeatureExtractor(object):
    def __init__(self):
        pass


    def fit(self, X_df, y_array):
        pass


    def transform(self, X_df):
        def haversine(lon1, lat1, lon2, lat2):
            """
            Calculate the great circle distance between two points 
            on the earth (specified in decimal degrees)
            """
            # convert decimal degrees to radians 
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

            # haversine formula 
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a)) 
            r = 6371 # Radius of earth in kilometers. Use 3956 for miles
            return c * r
        X_encoded = X_df.copy()
        path = os.path.dirname(__file__)
        data_weather = pd.read_csv(os.path.join(path, 'external_data.csv'))


         #Weather
        X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC']]
        X_weather = X_weather.rename(
            columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
        
        X_encoded = pd.merge(
            X_encoded, X_weather, how='left',
            left_on=['DateOfDeparture', 'Arrival'],
            right_on=['DateOfDeparture', 'Arrival'],
            sort=False)
        
        #External data but not weather
        X_ext=data_weather[['Date', 'AirPort','Rank_2018','State','city','lat','lng','population','density','ranking','Fuel_price','Holiday']]
        
        Dep_data=X_ext.add_suffix('_Dep')
        Arr_data=X_ext.add_suffix('_Arr')
        
        X_encoded=pd.merge(X_encoded, Dep_data, how='left', left_on=['DateOfDeparture', 'Departure'],
            right_on=['Date_Dep', 'AirPort_Dep'],
            sort=False )
        
        X_encoded=pd.merge(X_encoded, Arr_data, how='left', left_on=['DateOfDeparture', 'Arrival'],
            right_on=['Date_Arr', 'AirPort_Arr'],
            sort=False )
        
        
        #Dummy Depart/Arr
        X_encoded = X_encoded.join(pd.get_dummies(
            X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(
            pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        

        
        #Distance calculation
        X_encoded['Distance']=X_encoded.apply(lambda x:
        haversine(x['lng_Dep'],x['lat_Dep'],x['lng_Arr'],x['lat_Arr']),axis=1)
        
        
        #Dummy Date
        X_encoded['Weekend'] = ((pd.DatetimeIndex(X_encoded['DateOfDeparture']).dayofweek) // 5 == 1).astype(float)
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

        #We one hot encode all those
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        
        X_encoded=X_encoded.drop(['Date_Dep','Date_Arr',
                                  'city_Dep','city_Arr','AirPort_Dep',
                                  'State_Dep', 'AirPort_Arr', 'State_Arr',
                                  'Fuel_price_Arr','Holiday_Arr','year','month','week','day',
                                  'weekday','lat_Dep','lng_Dep','lat_Arr','lng_Arr'], axis=1)
        return X_encoded    
