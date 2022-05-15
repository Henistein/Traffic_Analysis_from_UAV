import pandas as pd
from datetime import datetime

names = ["altitude_above_seaLevel(feet)", " compass_heading(degrees)", "latitude", "longitude", "altitude(feet)"," roll(degrees)", " pitch(degrees)", "height_above_takeoff(feet)", "gimbal_heading(degrees)", "gimbal_pitch(degrees)"]
data = pd.read_csv('May-4th-2022-06-22PM-Flight-Airdata.csv')

limit = [datetime.strptime("17:25:00", "%H:%M:%S"),
         datetime.strptime("17:29:43", "%H:%M:%S")]

# find the begin index and end index
b,e = 0,0
for i,row in enumerate(data["datetime(utc)"]):
  s = row.split(' ')[1]
  d = datetime.strptime(s, "%H:%M:%S")
  if d >= limit[0] and b == 0:
    b = i
  if d >= limit[1] and e == 0:
    e = i

filtered_data = data.loc[b:e]
filtered_data = filtered_data[names]
filtered_data.to_csv('filtered_data.csv')
