from stravalib import Client
import webbrowser, pickle
#import urllib3.contrib.pyopenssl
from urllib2 import urlopen
from json import load, dumps
import thresher
import pandas as pd
import datetime
from pandas import DataFrame, pivot_table
import numpy as np

#urllib3.contrib.pyopenssl.inject_into_urllib3()

ACCESS_TOKEN = '1a253e9fdc49c806f1c052d791b7e26ebeb04a8b'

client = Client(access_token=ACCESS_TOKEN)

athlete = client.get_athlete() # Get current athlete details

print(athlete.firstname)

activity = client.get_activity_streams(438441582)

activities = client.get_activities()

sage = client.get_athlete(1595767)

print(sage.firstname)
print(sage.city)

print(athlete.id)

activities = list(client.get_activities())
"""
df = thresher.masterAssemble(client) 
oFile = open("master_dfs/"+str(athlete.id)+"masterDf.txt", 'w')
pickle.dump(df, oFile)
oFile.close()  
"""


inFile = open("master_dfs/"+str(athlete.id)+"masterDf.txt", 'r')
df = pickle.load(inFile)
inFile.close()

r = df[df.activityId == 466777343].loc[:,['latlng', 'hr', 'speeds', 'speedDelta3Abs', 'altDeltas', 'time', 'distCum']]
r_json = r.to_json(orient="records")

print(r_json)
"""


for i in range(len(activities)):
    if float(activities[i].id) not in list(df.activityId):
        print(activities[i].id)

#print(type(activities[1].id))
####################################################
# Summary Df
summaryDf = thresher.getSummaryDf(df)
#print(summaryDf.to_json(orient="records"))


 
recovery = DataFrame({'x':summaryDf.date, 'y':summaryDf.recovery})
recovery = recovery.set_index(np.arange(len(recovery)))
recovery_json = recovery.to_json(orient="index") # "index" used for stacked specific layout. 
print(recovery_json)

easy = DataFrame({'x':summaryDf.date, 'y':summaryDf.easy})
easy = easy.set_index(np.arange(len(easy)))
easy_json = easy.to_json(orient="index")
print(easy_json)

stamina = DataFrame({'x':summaryDf.date, 'y':summaryDf.stamina})
stamina = stamina.set_index(np.arange(len(stamina)))
stamina_json = stamina.to_json(orient="index")
print(stamina_json)

impulse = DataFrame({'x':summaryDf.date, 'y':summaryDf.impulse})
impulse = impulse.set_index(np.arange(len(impulse)))
impulse_json = impulse.to_json(orient="index")
print(impulse_json)


############################################################
# rolling Df
rollDf = thresher.getRollingSummaryDf(summaryDf)

rollDf =rollDf.fillna(0)

rollRec = DataFrame({'x':rollDf.date, 'y':rollDf.rollRec})
rollRec["y"] = rollRec.y + 100
rollRec = rollRec.set_index(np.arange(len(rollRec)))
rollRec_json = rollRec.to_json(orient="index")
print("rec", rollRec_json)

rollEasy = DataFrame({'x':rollDf.date, 'y':rollDf.rollEasy})
rollEasy = rollEasy.set_index(np.arange(len(rollEasy)))
rollEasy_json = rollEasy.to_json(orient="index")
print("easy", rollEasy_json)

rollStam = DataFrame({'x':rollDf.date, 'y':rollDf.rollStam})
rollStam = rollStam.set_index(np.arange(len(rollStam)))
rollStam_json = rollStam.to_json(orient="index")
print("stam", rollStam_json)

rollImp = DataFrame({'x':rollDf.date, 'y':rollDf.rollImp})
rollImp = rollImp.set_index(np.arange(len(rollImp)))
rollImp_json = rollImp.to_json(orient="index")
print("imp", rollImp_json)


calendarCsv = pd.read_csv("/home/rudebeans/Desktop/intermediate-d3-master/calendarData.csv")
cal_json = calendarCsv.to_json(orient="records")
print(cal_json)



#print(rolling.to_json(orient='records'))    # 'records' used for normal data

#summaryDF.to_csv('stravaSummaryDf.csv')

#print(summaryDF.to_json(orient="records"))

"""

