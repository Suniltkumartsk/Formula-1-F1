import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import requests
import json 
import warnings

warnings.filterwarnings('ignore')

def fetch_race_data(year):
    url=f"http://ergast.com/api/f1/{year}/races.json"
    try:
        response=requests.get(url)
        data=response.json()
        races=data['MRData']['RaceTable']['Races']
        return races
    except Exception as e:
        print(f"Error fetching data for year {year}: {e}")
        return [] 
    

def fetch_race_results(year,round_num):
    url=f"http://ergast.com/api/f1/{year}/{round_num}/results.json"
    try:
        response=requests.get(url)
        data=response.json()
        results=data['MRData']['RaceTable']['Race'][0]['Results']
        return results
    
    except Exception as e:
        print(f"Error fetching the results for year and round {year} and {round_num}: {e}")
        return []
    
def fetch_qualifying_results(year,round_num):
    url=f"http://ergast.com/api/f1/{year}/{round_num}/qualifying.json"
    
    try:
        response=request.get(url)
        data=response.jason()
        result=data['MRData']['RaceTable']['Race'][0]['QualifyingResults']
        return result
    except Exception as e:
        print(f"Error in fecthing the qyualifying reauls for year {year} and round_num {round_num}: {e}")
        return []
    
    
def fetch_driver_standings(year):
    url=f"http://ergast.com/api/f1/{year}/driverstandings.json"
    try:
        response=requests.get(url)
        data=response.json()
        standings=data['MRData']['StandingTable']['StandingsLists'][0]['DriverStandings']
        return standings
    
    except Exception as e:
        print(f"Error fetching the driver standing for year {year}: {e}")
        return []
    
print("\nFetching data from Ergast API (this may take 2-3 minutes)...")
years=list(range(2018,2025))

all_race_data=[]

for year in years:
    print(f"  Fetching the {year}....",end=" ")
    races =fetch_race_data(year)
    
    for race in races[:len(races)]:
        round_num=race['round']
        race_name=race['name']
        
        
        results=fetch_race_results(year,round_num)
        qualifying_results=fetch_qualifying_results(year,round_num)
        driver_standings=fetch_driver_standings(year)
        
        if results:
            for result in results:
                driver_id=result['Driver']['driverId']
                driver_name=f"{result['Driver']['givenName']} {result['Driver']['familyName']}"
                constructor=result['Constructor']['name']
                position =result['position']
                points=result['points']
                grid=result['grid']
                
                
                quali_pos=None
                for quali in qualifying_results:
                    if quali['Driver']['driverId']==driver_id:
                        quali_pos=int(quali['position'])
                        break
                    
                standings_points=None
                for standing in driver_standings:
                    if standing['Driver']['driverId']==driver_id:
                        standings_points=int(standing['points'])
                        break
                    
                
                all_race_data.append({
                    'Year':int(year),
                    'Round':int(round_num),
                    'RaceName':race_name,
                    'DriverID':driver_id,
                    'DriverName':driver_name,
                    'Constructor':constructor,
                    'FinishPosition':int(position) if position != '\\N' else None,
                    'GridPosition': int(grid) if grid != '0' else None,
                    'QualifyingPosition': quali_pos,
                    'RacePoints': int(points) if points != '0' else 0,
                    'PreRacePoints': standings_points
                })
        
        print(f"({len(results)} Races)")
    
print(f"Total Recorded Races: {len(all_race_data)}")

df=pd.DataFrame(all_race_data)

df=df.dropna(subset=['FinishPosition','GridPosition','QualifyingPosition'])
print(f"Total Record After cleaning: {len(df)}")


# Feature Engineering

df=df.sort_values(['DriverID','Year','Round'])

df['Last5FinishAvg']=df.groupby('DriverID')['FinishPosition'].transform(
    lambda x: x.rolling(window=5,min_periods=1).mean()
)

df['CareerRaces']=df.groupby('DriverID').cumcount()+1
df['CareerWin']=df.groupby('DriverID')['FinishPosition'].transform(
    lambda x: (x==1).cumsum()
)

df['WinRate']=df['CareerWin']/df['CareerRaces']
df['GridImprovement']=df['GridPosition']-df['FinishPosition']

#if they did not finish  like DNF
df['Finished']=1

df = df.sort_values(['Year', 'Round'])

df['ConstructorAvgPosition'] = (
    df.groupby(['Year', 'Constructor'])['FinishPosition']
      .transform(lambda x: x.expanding().mean())
)

df['CurrentSeasonPointsPerRace'] = (
    df.groupby(['Year', 'DriverID']).cumsum()['RacePoints'] /
    (df.groupby(['Year', 'DriverID']).cumcount() + 1)
)

#Continue with EDA and Modeling
                    
        
    

