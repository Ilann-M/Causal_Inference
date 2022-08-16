import numpy as np
import pandas as pd
from datetime import date

def  Import_df (discovery, store = None ): 
    
    train_df = pd.read_csv("../Data/marketing.csv")
    train_df.Date = pd.to_datetime(train_df.Date)
    train_df.set_index('Date', inplace=True)
    
    if store is not None :
        train_df = train_df.query(f'Store == {store}')

    if discovery :
        train_df = train_df.drop(['Store'], axis=1)
        train_df.loc[train_df.StateHoliday != '0' , 'StateHoliday'] = 1.
        train_df.loc[train_df.StateHoliday == '0' , 'StateHoliday'] = 0.
        var_names = train_df.columns
        for variable in var_names :
            train_df[f'{variable}'] = train_df[f'{variable}'].astype('float')
    
    else : 
        train_df.loc[train_df.StateHoliday != '0' , 'StateHoliday'] = 1
        train_df.loc[train_df.StateHoliday == '0' , 'StateHoliday'] = 0
        train_df['StateHoliday'] = train_df['StateHoliday'].astype('int')
        
        train_df.loc[train_df.Promo != 0 , 'Promo'] = True
        train_df.loc[train_df.Promo == 0 , 'Promo'] = False
        train_df['Promo'] = train_df['Promo'].astype('bool')
     
    jour = ['lundi','mardi','mercredi','jeudi','vendredi','samedi','dimanche']
    for i in range (7):
        train_df.loc[train_df.DayOfWeek == i+1 , 'DayOfWeek'] = jour[i]
    
    return train_df