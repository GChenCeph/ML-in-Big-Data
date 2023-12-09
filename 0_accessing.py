# pip install jdcal

import sqlite3
import pandas as pd
import jdcal
from datetime import datetime


state_to_id = {
    "AL": 1, "AZ": 2, "AR": 3, "CA": 4, "CO": 5,
    "CT": 6, "DE": 7, "FL": 8, "GA": 9, "ID": 10,
    "IL": 11, "IN": 12, "IA": 13, "KS": 14, "KY": 15,
    "LA": 16, "ME": 17, "MD": 18, "MA": 19, "MI": 20,
    "MN": 21, "MS": 22, "MO": 23, "MT": 24, "NE": 25,
    "NV": 26, "NH": 27, "NJ": 28, "NM": 29, "NY": 30,
    "NC": 31, "ND": 32, "OH": 33, "OK": 34, "OR": 35,
    "PA": 36, "RI": 37, "SC": 38, "SD": 39, "TN": 40,
    "TX": 41, "UT": 42, "VT": 43, "VA": 44, "WA": 45,
    "WV": 46, "WI": 47, "WY": 48, "HI": 49, "AK": 50
}


def julian_to_gregorian(julian_date):
    year, month, day, frac = jdcal.jd2gcal(jdcal.MJD_0, julian_date - jdcal.MJD_0)
    return datetime(year, month, day)


conn = sqlite3.connect("FPA_FOD_20170508.sqlite")
df = pd.read_sql_query("SELECT STATE, DISCOVERY_DATE FROM Fires", conn) #"SELECT * FROM 'Fires'",conn

df['DISCOVERY_DATE'] = df['DISCOVERY_DATE'].apply(julian_to_gregorian)

df['year'] = df['DISCOVERY_DATE'].dt.year
df['month'] = df['DISCOVERY_DATE'].dt.month

df = df.drop('DISCOVERY_DATE', axis=1)

df['STATE'] = df['STATE'].map(state_to_id)

df = df.dropna(subset=['STATE'])

df['STATE'] = df['STATE'].astype(int)

# Rename the STATE column to state
df.rename(columns={'STATE': 'state'}, inplace=True)

df.to_csv("data.csv", index=False)