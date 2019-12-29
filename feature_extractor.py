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
            # Function to compute the great circle distance between 2 points on Earth
            
            # a. Convert decimal degrees (latitude & longitude) to radians 
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

            # b. Haversine formula
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a)) 
            r = 6371 # Earth radius in kilometers
            return c * r
        
        # 1) Reading data:

        X_encoded = X_df.copy()
        path = os.path.dirname(__file__)
        external_data = pd.read_csv(os.path.join(path, 'external_data.csv'))
        
        # 2) Starting with weather data transformation and merging with X_encoded:

        weather_data = external_data[['Date', 'AirPort', 'Mean TemperatureC', 'MeanDew PointC', 
                                         'Mean Humidity', 'Mean Sea Level PressurehPa', 'Mean VisibilityKm',
                                         'Mean Wind SpeedKm/h', 'CloudCover']]
        weather_data = weather_data.rename(columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})

        X_encoded = pd.merge(X_encoded, weather_data, how = 'left',
            left_on = ['DateOfDeparture', 'Arrival'],
            right_on = ['DateOfDeparture', 'Arrival'],
            sort = False)
        
        # 3) Airport data transformation and merging:

        airport_data = external_data[['Date', 'AirPort','Rank_2018','State','city','lat',
                                            'lng','population','density','Fuel_price','Holiday',
                                            'LOAD_FACTOR','2018_freq']]
    
        ## We need to distinguish data related to departure and arrival airports
        Dep_data = airport_data.add_suffix('_Dep')
        Arr_data = airport_data.add_suffix('_Arr')
        
        ## We then merge both departure and arrival information with X_encoded
        X_encoded = pd.merge(X_encoded, Dep_data, how = 'left', 
            left_on = ['DateOfDeparture', 'Departure'],
            right_on = ['Date_Dep', 'AirPort_Dep'],
            sort = False)
        
        X_encoded = pd.merge(X_encoded, Arr_data, how = 'left', 
            left_on = ['DateOfDeparture', 'Arrival'],
            right_on = ['Date_Arr', 'AirPort_Arr'],
            sort = False)
        
        ## Distance calculation
        X_encoded['Distance'] = X_encoded.apply(lambda x: 
            haversine(x['lng_Dep'], x['lat_Dep'], x['lng_Arr'], x['lat_Arr']), axis = 1)

        # 4) Flight route data transformation and merging:
        
        ## Add flight route information in X_encoded
        X_encoded['ROUTE'] = X_encoded[['Departure', 'Arrival']].apply(''.join, axis=1)

        ## Select flight route data in external dataset
        route_data = external_data[['Year_route', 'Quarter_route','daily_passengers', 
                                                'average_fare', 'ROUTE', 'AIR_TIME_MEAN']]
        
        ## Split date information in X_encoded to help for merging
        X_encoded['Weekend'] = ((pd.DatetimeIndex(X_encoded['DateOfDeparture']).dayofweek) // 
            5 == 1).astype(float)
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['quarter'] = X_encoded['DateOfDeparture'].dt.quarter
        X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date:
            (date - pd.to_datetime("1970-01-01")).days)
        
        ## Merge flight route data with X_encoded
        X_encoded = pd.merge(X_encoded, route_data, how = 'left', 
            left_on = ['year','quarter','ROUTE'],
            right_on = ['Year_route','Quarter_route','ROUTE'],
            sort = False)
    
        # 5) One-hot encoding of categorical features (Dep/Arr airports, dates):

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix = 'd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix = 'a'))
           
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['quarter'], prefix='q'))

        X_encoded['Rank_2018_Arr'] = X_encoded['Rank_2018_Arr'].astype(int)
        X_encoded['Rank_2018_Dep'] = X_encoded['Rank_2018_Dep'].astype(int)
                
        # 5) Final feature selection:
        ## We drop duplicated (one-hot encoded) and unsignificant features (based on feature importance analysis)
        
        X_encoded = X_encoded.drop(['DateOfDeparture','Departure','Arrival','Quarter_route',
                                    'Date_Dep','Date_Arr','city_Dep','city_Arr','AirPort_Dep',
                                    'State_Dep', 'AirPort_Arr', 'State_Arr','Fuel_price_Arr',
                                    'Holiday_Arr','year','month','week','day','weekday','lat_Dep',
                                    'lng_Dep','lat_Arr','lng_Arr','ROUTE','std_wtd','Year_route',
                                    'LOAD_FACTOR_Arr','Mean Sea Level PressurehPa','MeanDew PointC',
                                    'Mean Wind SpeedKm/h','Mean VisibilityKm', 'quarter',
                                    'w_32', 'w_18', 'w_33', 'w_19', 'w_31', 'w_24', 'd_21', 'd_16', 'w_28',
                                    'w_12', 'd_26', 'd_14', 'w_40', 'a_IAH', 'w_43', 'd_6', 'w_16', 'w_11',
                                    'w_17', 'd_22', 'w_30', 'w_29', 'd_19', 'd_17', 'w_23', 'd_SEA',
                                    'a_PHL', 'd_MSP', 'w_45', 'd_DEN', 'w_34', 'd_DFW', 'CloudCover',
                                    'a_SEA', 'w_13', 'Mean Humidity', 'd_20', 'd_11', 'w_25', 'd_15',
                                    'd_13', 'w_15', 'w_42', 'a_ATL', 'd_24', 'w_48', 'w_39',
                                    'y_2012', 'm_7', 'm_2', 'q_4'], axis=1)
        return X_encoded