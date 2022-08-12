import numpy as np
import pandas as pd
from datetime import date
import sklearn
import warnings
from sklearn.exceptions import DataConversionWarning 
from matplotlib import pyplot as plt

from dowhy import CausalModel

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

def CausAnal (complet_df, traitement, outcome, common_c=None, effect_mo=None, instruments=None, methode = None, t_binaire=True):
    
    common_c = [] if common_c is None else common_c
    effect_mo = [] if effect_mo is None else effect_mo
    instruments = [] if instruments is None else instruments

    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    
    model=CausalModel(data = complet_df, treatment= traitement, outcome=outcome, 
                      common_causes=common_c, effect_modifiers= effect_mo, instruments=instruments) 
    model.view_model() # return
    
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    if methode = None : 
        causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        print("Linear_regression Causal Estimate is " + str(causal_estimate.value) + "\n ") 
        
        if t_binaire :  
            causal_estimate = model.estimate_effect(identified_estimand,
                                                    method_name="backdoor.distance_matching")
            print("Distance_matching Causal Estimate is " + str(causal_estimate.value) + "\n ") 

            causal_estimate = model.estimate_effect(identified_estimand,
                                                    method_name="backdoor.propensity_score_stratification")
            print("Propensity_score_stratification Causal Estimate is " + str(causal_estimate.value) + "\n ") 

            causal_estimate = model.estimate_effect(identified_estimand,
                                                    method_name="backdoor.propensity_score_matching")
            print("Propensity_score_matching Causal Estimate is " + str(causal_estimate.value) + "\n ") 

            causal_estimate = model.estimate_effect(identified_estimand,
                                                    method_name="backdoor.propensity_score_weighting")
            print("Propensity_score_weighting Causal Estimate is " + str(causal_estimate.value) + "\n ") 

    else :
        if "ps" in methode and t_binaire :
            if 'strat' in methode :
                causal_estimate = model.estimate_effect(identified_estimand,
                                                    method_name="backdoor.propensity_score_stratification")
                print("Propensity_score_stratification Causal Estimate is " + str(causal_estimate.value) + "\n ")
            
            if 'match' in methode :
                causal_estimate = model.estimate_effect(identified_estimand,
                                                    method_name="backdoor.propensity_score_matching")
                print("Propensity_score_matching Causal Estimate is " + str(causal_estimate.value) + "\n ") 
               
            if "weight" in methode : 
                causal_estimate = model.estimate_effect(identified_estimand,
                                                    method_name="backdoor.propensity_score_weighting")
                print("Propensity_score_weighting Causal Estimate is " + str(causal_estimate.value) + "\n ") 
        else : 
            causal_estimate = model.estimate_effect(identified_estimand,
                                                    method_name="backdoor.distance_matching")
            print("Distance_matching Causal Estimate is " + str(causal_estimate.value) + "\n ")
            
################################ refutation 

    refute_random = model.refute_estimate(identified_estimand, causal_estimate, method_name="random_common_cause")
    print(refute_random)

    refute_placebo = model.refute_estimate(identified_estimand, causal_estimate,
            method_name="placebo_treatment_refuter", placebo_type="permute")
    print(refute_placebo)

    refute_subset = model.refute_estimate(identified_estimand, causal_estimate,
            method_name="data_subset_refuter", subset_fraction=0.9)
    print(refute_subset)

    refute_unobserved = model.refute_estimate(identified_estimand, causal_estimate, method_name="add_unobserved_common_cause",
                                         confounders_effect_on_treatment="binary_flip", confounders_effect_on_outcome="linear",
                                        effect_strength_on_treatment=0.01, effect_strength_on_outcome=0.02)
    print(refute_unobserved)

