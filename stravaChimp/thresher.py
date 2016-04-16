from pandas import DataFrame, Series, pivot_table
import numpy as np
import pickle, copy, datetime
import pandas as pd
import os
import statsmodels.api as sm
from geopy.distance import great_circle
#from models import Activity, Athlete



# adds a columns with accurate distances measured using great circle. Strava dists suck

def addDistDeltas(df): 

    df = basicClean(df)
    #df['latlng2'] = df.latlng.shift(1)
    df['lat'] = [z[0] for z in df['latlng']]
    df['lng'] = [z[1] for z in df['latlng']]
    df['lat2'] = df.lat.shift(1)
    df['lng2'] = df.lng.shift(1)
    #df['dist_r'] = np.sqrt((df.lat - df.lat2)**2 + (df.lng - df.lng2)**2)
    #print df[['lat', 'lng', 'lat2','lng2', 'dist_r']]
    
    df['distDeltas_old'] = copy.deepcopy(df.distDeltas) # saving old distances just in case
    df['speeds_old'] = copy.deepcopy(df.speeds)
    
    df['distDeltas'] = np.nan
    for i in range(len(df)):
        df.ix[df.index[i], 'distDeltas'] = great_circle((df.lat[df.index[i]], df.lng[df.index[i]]), (df.lat2[df.index[i]], df.lng[df.index[i]])).meters

    df['speeds'] = df.distDeltas / df.timeDeltas
    
    df = df.fillna(value=0.0)
    #df = df[['date', 'distDeltas', 'speeds']] 
    
    #df = df.loc[1:len(df)]
    
    #print df
    
    return df
    
    
    
    
# assembling data into usable df
def assemble(date, activityId, hrList, distanceList, timeList, altitudeList, latlngList, cadence):
    
    """
    df = DataFrame({})
    df['latlng'] = latlngList
    df['hr'] = hrList
    df['time'] = timeList
    df['altitude'] = altitudeList
    df['cadence'] = cadence
    """
   
    speeds = []
    timeDeltas = []
    distDeltas = []
    altDeltas = []
    for i in range(1,len(timeList)):
        timeSegment = float(timeList[i] - timeList[i-1])
        distanceCovered = float(distanceList[i] - distanceList[i-1])
        speedEntry = distanceCovered / timeSegment
        altDeltaEntry = float(altitudeList[i] - altitudeList[i-1])       
        altDeltas.append(altDeltaEntry)
        speeds.append(speedEntry)
        timeDeltas.append(timeSegment)
        distDeltas.append(distanceCovered) 
                  
    hrL = hrList[1:len(hrList)]     
    latlngL = latlngList[1:len(latlngList)]  
    timeL = timeList[1:len(timeList)]
    cadence = cadence[1:len(cadence)]
    date = [date]*len(timeL)

    # assembling
    df = DataFrame({'date':date, "latlng":latlngL, "activityId":activityId, 'hr': hrL, 'speeds':speeds, 'time': timeL, 'timeDeltas':timeDeltas, 'distDeltas':distDeltas, 'altDeltas':altDeltas, 'cadence':cadence}) 
     
    # adding smoothed speeds
    df['speedsSmoothed2'] = pd.rolling_mean(df.speeds,2)
    df['speedsSmoothed3'] = pd.rolling_mean(df.speeds,3)
    
    # adding percentange AND absolute speed changes
    df['speedDelta2'] = df.speeds.pct_change(periods=2)    
    df['speedDelta3'] = df.speeds.pct_change(periods=3)  
    s = [np.nan]*3 + list(df.speeds[:-3])
    df['speedDelta3Abs'] = df.speeds - s   
    df['speedDelta5'] = df.speeds.pct_change(periods=5)
    df['speedDelta8'] = df.speeds.pct_change(periods=8)
    
    # Calculating a normalizer column (a scale, i.e. hr 145 = 1.0) to yield load / Real Miles calculation. 
    easyHr = 145
    minHr = 80
    threshhold = 170     
    # Defining a function to be mapped onto df.hr
    def normalizer(x):
        if (x <= easyHr) and (x >= minHr):
            slope = 1.0 / (easyHr - minHr)
            a = -slope*minHr
            normalizer = a + slope*x # a number centered on 1
            x = normalizer
            return x
        if (x > easyHr):
            slope = 1.0 / (threshhold - easyHr) 
            a = -slope*easyHr
            normalizer = a + slope*x + 1
            x = normalizer
            return x      
    # mapping      
    normalizer = df.hr.map(normalizer)
    df['normalizer'] = normalizer

    # Real Dist / normalized load 
    df['realDistInd'] = df.normalizer * df.distDeltas
    df['realDistCum'] = np.cumsum(df.realDistInd)    
    df['distCum'] = np.cumsum(df.distDeltas)    
       
    return df

#df = assemble(date, hr, distance, time, altitude) 
#print(df)

# takes in client and outputs master df with all activities appended end to end. It checks to see which activities have already been uploaded, only appending new entries
def masterAssemble(client):
    activities = list(client.get_activities())
    print(len(activities))
    athlete = client.get_athlete()
    # add in name of run

    path = os.path.dirname(__file__)
    
    try: 
        df = pd.read_pickle(str(path)+"/master_dfs/"+str(athlete.id)+"masterDf.txt")
    except IOError:
        df = DataFrame({})
    
    for i in range(len(activities)):
        if (len(df) == 0) or (float(activities[i].id) not in list(df.activityId)):
            activityId = activities[i].id
            run = client.get_activity_streams(activityId, types=['time','latlng','distance','heartrate','altitude',])
            latlng = run['latlng'].data
            time = run['time'].data
            distance = run['distance'].data
            heartrate = run['heartrate'].data
            altitude = run['altitude'].data
            date = activities[i].start_date_local 
            activity = activityId   
            dfi = assemble(date, activityId, heartrate, distance, time, altitude, latlng)
            df = df.append(dfi)    
            print(dfi)          
    return df
     

# making summary df
def getSummaryDf(df):
    sumDf = pivot_table(df, ['timeDeltas','realDistInd', 'distDeltas'], 'date', aggfunc=np.sum) 
    dates = sumDf.index
    summaryDf = DataFrame({})
    for i in range(len(dates)):
        run = df[df.date == dates[i]]
        climb = getClimb(run)
        recovery = getRecovery(run)
        #easySpeed = np.round(getEasySpeed(run),2)# fitness proxy. fix this
        totalDist = np.sum(run.distDeltas)
        realMiles = np.round(np.sum(run.realDistInd), 0)
        totalTime = np.round(np.sum(run.timeDeltas), 0)
        dateTime = dates[i]
        date = datetime.datetime(dateTime.year, dateTime.month, dateTime.day)
        variation = np.round(getHrVar(run), 0)
        avgHr = np.round(getHrAvg(run), 0)    
        activityId = run.activityId[0]
        summary = DataFrame({'date':date,'activityId':activityId, 'avgHr':avgHr,'realMiles':realMiles, 'totalDist':totalDist, 'totalTime':totalTime, 'variation': variation, 'recovery':recovery, 'climb':climb, 'impulse':getImpulse(run), 'stamina':getStamina(run), 'easy':getEasy(run), 'recovery':getRecovery(run)}, index=[date])
        summaryDf = summaryDf.append(summary)        
    return(summaryDf)
    
# making SINGLE RUN summary df
def getSingleSummaryDf(run):

    climb = getClimb(run)
    recovery = getRecovery(run)
    #easySpeed = np.round(getEasySpeed(run),2)# fitness proxy. fix this
    totalDist = np.sum(run.distDeltas)
    realMiles = np.round(np.sum(run.realDistInd), 0)
    totalTime = np.round(np.sum(run.timeDeltas), 0)
    dateTime = run.date[0]
    #date = datetime.datetime(dateTime.year, dateTime.month, dateTime.day)
    
    
    variation = np.round(getHrVar(run), 0)
    avgHr = np.round(getHrAvg(run), 0)    
    activityId = run.activityId[0]
    summary = {'date':dateTime,'activityId':activityId, 'avgHr':avgHr,'realMiles':realMiles, 'totalDist':totalDist, 'totalTime':totalTime, 'variation': variation, 'recovery':recovery, 'climb':climb, 'impulse':getImpulse(run), 'stamina':getStamina(run), 'easy':getEasy(run), 'recovery':getRecovery(run)}
                       
    return(summary)

# making df for rolling figures
def getRollingSummaryDf(summaryDf):
    start = min(summaryDf.index)
    end = datetime.datetime.now()
    dates = DataFrame({'date':pd.date_range(start, end)})
    fullSummary = pd.merge(dates, summaryDf, how='outer',on=['date'])    
        
    rollingDistL = []   
    rollingImpulseL = []
    rollingRecL = [] 
    rollingEasyL = []
    rollingStaminaL = []  
    for i in range(len(fullSummary)):
        aWeekPrevious = fullSummary.date[i] - datetime.timedelta(days=7)
        previous7Days = fullSummary[(fullSummary.date > aWeekPrevious) & (fullSummary.date <= fullSummary.date[i])]
        
        rollingDist = np.sum(previous7Days.totalDist)
        rollingDistL.append(rollingDist)

        rollingImpulse = np.sum(previous7Days.impulse)
        rollingImpulseL.append(rollingImpulse)
        
        rollingRec = np.sum(previous7Days.recovery)
        rollingRecL.append(rollingRec)
        
        rollingEasy = np.sum(previous7Days.easy)
        rollingEasyL.append(rollingEasy)
        
        rollingStamina = np.sum(previous7Days.stamina)
        rollingStaminaL.append(rollingStamina)

    rolling = DataFrame({'date':fullSummary.date, 'rollDist':rollingDistL, 'rollImp':rollingImpulseL, 'rollRec':rollingRecL, 'rollEasy':rollingEasyL, 'rollStam':rollingStaminaL})
    
    return rolling

# making json for testing purposes
#testJson = df.loc[:,['hr','speeds', 'altDeltas', 'speedDelta3', 'speedDelta3Abs', 'speedsSmoothed2', 'realDistCum', 'time']].to_json(orient="records")
#print(testJson)

def histogram(df): # returns histogram of time spent at each hr
     hist = pivot_table(df, ['timeDeltas'], 'hr', aggfunc=np.sum)
     hist['hr'] = hist.index
     hist['timeSmoothed'] = pd.rolling_mean(hist.timeDeltas, 2)
     return hist
     
#print(histogram(df).to_json(orient="records"))     
#fitLine = pivot_table(df2, ['speeds'], 'hr', aggfunc=np.mean) 

#hrTime_json = df.hr.to_json()

#openFile = open("hrTime_json.txt", 'w')
#pickle.dump(hrTime_json, openFile)
#openFile.close()


def getFitLine(df): # returns one-column df with avg speed by hr (index)
    df2 = copy.deepcopy(df)
    counter = 1
    while counter < len(df2)-6:
        diff = np.sqrt((df2.speeds[counter] - df2.speeds[counter-1])**2) 
        percentDiff = diff / (df2.speeds[counter-1] + .00000001)    
        if percentDiff > .3:
            df2.ix[counter, 'speeds'] = np.nan
            df2.ix[counter-1, 'speeds'] = np.nan
            df2.ix[counter+1, 'speeds'] = np.nan
            df2.ix[counter+2, 'speeds'] = np.nan
            df2.ix[counter+3, 'speeds'] = np.nan
            df2.ix[counter+4, 'speeds'] = np.nan
            counter += 6
        else: counter += 1
    # only keeping entries where altDelta less than 1, speed greater than 1, distDelta greater than 1, and change in speed less than threshhold   
    df2 = df2[(np.sqrt(df2.altDeltas**2) < 1.0) & (df2.speeds > 0.5)]    
    # consolidating to hr groups
    hrGroups = np.arange(60,200,5) # buckets five wide
    def assignHrGroups(x):
        for i in range(len(hrGroups)-1):
            if x >= hrGroups[i] and x < hrGroups[i+1]:
                return hrGroups[i]                             
    hrs = df2.hr
    hrs = hrs.map(assignHrGroups)
    df2['hrGroup'] = hrs    
    # making pivot table to consolidate by hr
    fitLine = pivot_table(df2, ['speeds'], 'hrGroup', aggfunc=np.mean) 
    f = pivot_table(df2, ['speeds'], 'hrGroup', aggfunc=len) 
    
    fitLine = DataFrame({'date': df.date[0], 'hr':fitLine.index, 'avgSpeed':fitLine.speeds, 'count':f.speeds})   
    fitLine = fitLine[fitLine['count'] >= 10] # only keeping groups w at least 20 entries
    return fitLine
   
    
def getFitDf(df):  # returns df w fit-cleaned data
    df2 = copy.deepcopy(df)
    counter = 1
    while counter < len(df2)-6:
        diff = np.sqrt((df2.speeds[counter] - df2.speeds[counter-1])**2) 
        percentDiff = diff / (df2.speeds[counter-1] + .00000001)    
        if percentDiff > .3:
            df2.speeds[counter] = np.nan
            df2.speeds[counter-1] = np.nan
            df2.speeds[counter+1] = np.nan
            df2.speeds[counter+2] = np.nan
            df2.speeds[counter+3] = np.nan
            df2.speeds[counter+4] = np.nan
            counter += 6
        else: counter += 1
    # only keeping entries where altDelta less than 1, speed greater than 1, distDelta greater than 1, and change in speed less than threshhold        
    df2 = df2[(np.sqrt(df2.altDeltas**2) < 1.0) & (df2.speeds > 1.0)]    
    return df2
    


# cleaning up only the most outrageous observations
def basicClean(df):
    # dropping all observations where speed > 8
    df = df[df.speeds<8.0]

    # dropping all observations where altDeltas > 5
    df = df[df.altDeltas < 5.0]  
    return df
    
# takes in a df, adds predicted values for hr and speed
def getPred(df):
    
    # adding intercept
    df['intercept'] = 1
    
    # creating higher order, shifted, and interacted variables
    df['cadence2'] = df.cadence**2
    df['ln_cadence'] = np.log(df.cadence)
    df['hr_cadence'] = df.hr*df.cadence
    df['hr_altDeltas'] = df.hr*df.altDeltas
    df['altDeltas_cadence'] = df.altDeltas*df.cadence
    df['hr_shift1'] = df.hr.shift(-1)
    df['hr_shift2'] = df.hr.shift(-2)
    df['hr_shift3'] = df.hr.shift(-3)
    df['hr_shift4'] = df.hr.shift(-4)
    df['hr_next4'] = df.hr_shift1 * df.hr_shift2 * df.hr_shift3 * df.hr_shift4
    
    df['speed_shift1'] = df.speeds.shift(1)
    df['speed_shift2'] = df.speeds.shift(2)
    df['speed_shift3'] = df.speeds.shift(3)
    df['speed_shift4'] = df.speeds.shift(4)
    df['speed_shift5'] = df.speeds.shift(5)
    df['speed_shift6'] = df.speeds.shift(6)
    df['speed_shift7'] = df.speeds.shift(7)
    df['speed_shift8'] = df.speeds.shift(8)
    df['speed_shift9'] = df.speeds.shift(9)
    df['speed_shift10'] = df.speeds.shift(10)
    df['speed_shift11'] = df.speeds.shift(11)
    df['speed_shift12'] = df.speeds.shift(12)

    df['speeds2'] = df.speeds**2
    
    # creating copy before we trim df to use in prediction
    origDf = copy.deepcopy(df)
    
    # dropping all observations where cadence is zero
    df = df[df.cadence>0]

    # shifting creates NAs, dropping them
    df = df.dropna()

    # predicting speed, adding speedHat and speedRes to df
    X = df[['intercept', 'altDeltas', 'cadence', 'cadence2', 'hr','hr_shift1','hr_shift2','hr_shift3', 'hr_shift4', 'hr_cadence', 'ln_cadence']]
    lm = sm.OLS(df.speeds, X).fit()
    speedHat = lm.predict(X)
    df['speedHat'] = speedHat
    df['speedRes'] = np.abs(speedHat-df.speeds) / np.std(df.speeds)
    #print(df[['speedHat', 'speeds']])
    print(lm.summary())

    ###################################################
    # Predicting hr, same as above


    df = df.dropna()

    xCols = ['intercept', 'altDeltas', 'cadence', 'cadence2', 'ln_cadence', 'speeds', 'speeds2', 'speed_shift1', 'speed_shift2', 'speed_shift3', 'speed_shift4', 'speed_shift5','speed_shift6', 'speed_shift7', 'speed_shift8', 'speed_shift9', 'speed_shift10', 'speed_shift11', 'speed_shift12']
    X = df[xCols]
    lm = sm.OLS(df.hr, X).fit()
    df['hrHat'] = lm.predict(X)
    df['hrRes'] = np.abs(df.hrHat-df.hr) / np.std(df.hr)
    #print(df[['hr', 'hr_hat']])
    print(lm.summary())
    
    return df


# takes in a df, then adds lowess-smoothed points for plotting hr over time, and speed over time
def addLws(df):
    df = df.dropna()
    numPoints = 8.0
    fraction = numPoints / float(len(df))
    print(fraction)

    lowess = sm.nonparametric.lowess

    # lowess for speed
    lws = lowess(df.speeds, df.time, frac = fraction, it=0)
    lwsX = [z[0] for z in lws]
    lwsY = [z[1] for z in lws]

    df['speedLwsY'] = lwsY
    df['speedLwsX'] = lwsX

    # lowess for hr
    hrLws = lowess(df.hr, df.time, frac = fraction, it=0)
    hrLwsX = [z[0] for z in hrLws]
    hrLwsY = [z[1] for z in hrLws]

    df['hrLwsY'] = hrLwsY
    df['hrLwsX'] = hrLwsX

    return df
    
# takes in a df (must have been getPred()-ed in the past), outputs three-column df with date, hr points, and speed point
def getFitlineLws(df):
   
    print(len(df))
    #df = df[df.cadence>0]
    #df = df[(df.hrRes < 3.0) & (df.speedRes < 3.0)]

    df['speed_shift1'] = df.speeds.shift(1)
    df['speed_shift2'] = df.speeds.shift(2)
    df['speed_shift3'] = df.speeds.shift(3)
    df['speed_shift4'] = df.speeds.shift(4)
    df['speed_shift5'] = df.speeds.shift(5)
    df['speed_shift6'] = df.speeds.shift(6)
    df['speed_shift7'] = df.speeds.shift(7)
    df['speed_shift8'] = df.speeds.shift(8)
    df['speed_shift9'] = df.speeds.shift(9)
    df['speed_shift10'] = df.speeds.shift(10)
    df['speed_shift11'] = df.speeds.shift(11)
    df['speed_shift12'] = df.speeds.shift(12)


    threshhold = .25
    #print(len(df))
    for i in range(1,10):
        dif = 'speed_diff'+str(i)
        df[dif] = np.abs(df.speeds - df['speed_shift'+str(i)]) / df.speeds

    df = df[(df.speed_diff7 < threshhold) & (df.speed_diff6 < threshhold) & (df.speed_diff5 < threshhold) & (df.speed_diff5 < threshhold) & (df.speed_diff4 < threshhold) & (df.speed_diff3 < threshhold) & (df.speed_diff2 < threshhold) & (df.speed_diff1 < threshhold)]

    print(len(df))
    df = df[np.abs(df.altDeltas) < 2.0]
    print(len(df))
    

    lowess = sm.nonparametric.lowess
    #jitter = np.random.normal(0, .00001, len(df))
    #df['speeds'] = df.speeds + jitter # bug in statsmodels, adding some jitter as workaround

    lws = lowess(df.speeds, df.hr, frac = 1.0, it=0)
    lwsX = [z[0] for z in lws]
    lwsY = [z[1] for z in lws]

    flLws = pd.DataFrame({})
    flLws['hr'] = lwsX
    flLws['avgSpeed'] = lwsY
    flLws['date'] = df.date[df.index[0]]
    #print(flLws)
    
    return flLws
    
"""
#print(getFitLine(df))
df.loc[:,['hr','time']].to_csv("/home/rudebeans/jan10_project/calculator/static/calculator/hrTime.csv", index=False, header=True)

df.loc[:,['speeds','time']].to_csv("/home/rudebeans/jan10_project/calculator/static/calculator/speedsTime.csv", index=False, header=True)

getFitLine(df).to_csv("/home/rudebeans/jan10_project/calculator/static/calculator/fitLine.csv", index=False, header=True)

df.loc[:,['hr','speeds']].to_csv("/home/rudebeans/jan10_project/calculator/static/calculator/scatter.csv", index=False, header=True)

getFitDf(df).loc[:,['hr','speeds']].to_csv("/home/rudebeans/jan10_project/calculator/static/calculator/scatterFitCleaned.csv", index=False, header=True)
 """
 
def getHrVar(df):
    minHr = 80
    df = df[df.hr > minHr]
    hrVar = np.std(df.hr)
    return hrVar
    
def getHrAvg(df):
    minHr = 80
    df = df[df.hr > minHr]
    avg = np.mean(df.hr)
    return avg
    
def getClimb(df):
    df = df[df.altDeltas > 0.0]
    climb = np.sum(df.altDeltas)
    return climb
    
def getTotalDistance(df):
    totalDistance = np.sum(df.distDeltas)
    return totalDistance

#print(getTotalDistance(df))


minRecHr = 90
maxRecHr = 125
minLoadAcc = 126
minImpulseAcc = 165
maxHr = 190

def getLoad(df): # simple version 
    loadDf = df[df.hr >= minLoadAcc]
    totalLoad = np.sum(loadDf.hr * loadDf.timeDeltas)
    return totalLoad
  
# getting distance spent in each zone
impulse = 165
stamina = 145
easy = 120
recovery = 80    
def getImpulse(df):
    impulseDf = df[df.hr >= impulse]
    totImpulse = np.sum(impulseDf.distDeltas)
    return totImpulse
def getStamina(df):
    staminaDf = df[(df.hr >= stamina) & (df.hr < impulse)]
    totStamina = np.sum(staminaDf.distDeltas)
    return totStamina   
def getEasy(df):
    easyDf = df[(df.hr >= easy) & (df.hr < stamina)]
    totEasy = np.sum(easyDf.distDeltas)
    return totEasy
def getRecovery(df):
    recoveryDf = df[(df.hr >= recovery) & (df.hr < easy)]
    totRecovery = np.sum(recoveryDf.distDeltas)
    return totRecovery
def getZones(df):
    zones = {'impulse':getImpulse(df), 'stamina':getStamina(df), 'easy':getEasy(df), 'recovery':getRecovery(df)}
    return zones


sampleMin = 135 # simple fit score, ie speed at easy pace
sampleMax = 145    
def getEasySpeed(df): 
    fitLine = getFitLine(df)
    segment = fitLine[(fitLine.index >= sampleMin) & (fitLine.index <= sampleMax)]
    avgSpeed = np.mean(segment.avgSpeed)
    return avgSpeed


  

easyHr = 145 # stamina
minHr = 80 # recovery
threshhold = 170 # impulse
  
def getRealMiles(df): # Returns normalized distance, ie real load

    slope = 1.0 / (stamina - recovery)
    a = -slope*recovery
    easy = df[(df.hr <= stamina) & (df.hr >= recovery)]
    easy['normalizer'] = a + slope*(easy.hr)
    easy['normalizedDist'] = easy.normalizer * easy.distDeltas
    easyDist = np.sum(easy.normalizedDist)
    
    slope = 1.0 / (impulse - stamina) 
    a = -slope*stamina
    thresh = df[df.hr > stamina] # thresh = impulse
    thresh['normalizer'] = a + slope*(thresh.hr) + 1
    thresh['normalizedDist'] = thresh.normalizer * thresh.distDeltas
    threshDist = np.sum(thresh.normalizedDist)
    
    normalizedDist = threshDist + easyDist
    return normalizedDist
    





#print(getRecovery(df))
    
#print(df.hr)
    
"""
# TODO: change this to simpler absolute scale. Impulse calculations: Everything above minImpulseAcc counts as impulse. Time at max HR counted as 3 times more impulse accumulation than at minImpulseAcc. Normalized impulse is multiplied by time, then summed up for a total impulse figure

# Projecting onto the scale described directly above. Note magic number 2 in normalizedImpulse

def getImpulse(df):
    normalizerDomain = maxHr - minImpulseAcc
    impulseDf = df[df.hr >= minImpulseAcc]
    impulseDf['rawImpulse'] = impulseDf['hr'] - minImpulseAcc
    impulseDf['normalizedImpulse'] = 1.0 + (2.0 * impulseDf.rawImpulse) / normalizerDomain
    totalImpulse = np.sum(impulseDf.normalizedImpulse * impulseDf.timeDeltas)    
    return totalImpulse
"""

def getSpeed(distance, time):
    speed = list()
    for i in range(1,len(time)):
        timeSegment = time[i] - time[i-1]
        distanceCovered = distance[i] - distance[i-1]
        speedEntry = distanceCovered / timeSegment
        speed.append(speedEntry)
        
        
#print(getSpeed(distance, time))


