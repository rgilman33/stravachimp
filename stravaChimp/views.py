from django.shortcuts import render
import numpy as np
from urllib2 import urlopen, Request
from json import load, dumps
import requests, datetime
from django.http import HttpRequest 
import pickle, pandas
from pandas import DataFrame, Series
import pandas as pd
from stravalib import Client
#import urllib3.contrib.pyopenssl
#urllib3.contrib.pyopenssl.inject_into_urllib3()
import thresher, os
from models import Activity, Athlete

client = Client()

# Got from Strava after registering on website
MY_STRAVA_CLIENT_ID = "9558"
MY_STRAVA_CLIENT_SECRET = "734e394ff653703abaef7c7c061cc8d685241a10"

# a url sending visitor to strava's website for authentication
# THIS MUST BE UPDATED FOR WEB USE, strava website must be updated as well
#stravaURL = client.authorization_url(client_id=MY_STRAVA_CLIENT_ID, redirect_uri='http://reddlee.pythonanywhere.com/authorization')
stravaURL = client.authorization_url(client_id=MY_STRAVA_CLIENT_ID, redirect_uri='http://127.0.0.1:8000/authorization')

def index(request):
    return render(request, 'stravaChimp/index.html', {'c':stravaURL})
    
#
# visitors sent to this page after agreeing to allow our site to use their strava
def authorization(request):
    client = Client()
    code = request.GET['code']
    access_token = client.exchange_code_for_token(client_id=MY_STRAVA_CLIENT_ID, client_secret=MY_STRAVA_CLIENT_SECRET, code=code)   
    
    # making a global variable to be used across views. don't know how this will work in practice
    
    client = Client(access_token=access_token)
    athlete = client.get_athlete() # Get current athlete details
    
    global athleteId 
    athleteId = athlete.id
    
    # if athlete doesn't exist, add them
    if len(Athlete.objects.filter(athleteId=athleteId)) == 0:
        ath = Athlete.objects.create(name=str(athlete.firstname+' '+athlete.lastname), athleteId=athleteId, profilePic=athlete.profile, city=athlete.city, country=athlete.country, sex=athlete.sex, premium=athlete.premium, created_at=athlete.created_at, updated_at=athlete.updated_at, followers=athlete.follower_count, friends=athlete.friend_count, email=athlete.email, weight=athlete.weight, meas_pref=athlete.measurement_preference, runsSummary = DataFrame({}).to_json(orient='records'), fitLines = DataFrame({}).to_json(orient='records'), masterList = DataFrame({}).to_json(orient='records'))

        ath.profilePic.name = "rudyzPic"
        ath.save(update_fields=['profilePic'])
 
    # if athlete already exists, draw their file
    elif len(Athlete.objects.filter(athleteId=athleteId)) == 1:
        ath = Athlete.objects.get(athleteId=athleteId)
           
    ############################################ 
    ##### compiling new runs, updating summary
        
    # athlete's existing runs summary   
    existingSummary = DataFrame(pd.read_json(ath.runsSummary))
    existingFitlines = DataFrame(pd.read_json(ath.fitLines)) 
    masterList = DataFrame(pd.read_json(ath.masterList))
     
    activities = list(client.get_activities()) 
    
    # activity IDs of runs already in the system
    try:
        ids = existingSummary.activityId
    except AttributeError:
        ids = []
         
    for i in range(len(activities)):   
    #for i in range(30,37):
        # Ignoring activities already in the system 
        if (len(ids) == 0) or (float(activities[i].id) not in list(ids)):
            
            try:
                # compiling df for raw json-ization
                activityId = activities[i].id
                run = client.get_activity_streams(activityId, types=['time','latlng','distance','heartrate','altitude','cadence'])
                latlng = run['latlng'].data
                time = run['time'].data
                distance = run['distance'].data
                heartrate = run['heartrate'].data
                altitude = run['altitude'].data
                cadence = run['cadence'].data
                date = activities[i].start_date_local 
                activity = activityId   
                dfi = thresher.assemble(date, activityId, heartrate, distance, time, altitude, latlng, cadence) 
                
                
                # basic cleanup, only removing totally unreasonable values
                dfi = thresher.basicClean(dfi)


                # if we ever want to try our hand at improving strava's speed data (ie by predicting speed when GPS blanks), intervene here:
                    
                #dfi = thresher.addDistDeltas(dfi)
                             
                                        
                try: 
                    fitline = thresher.getFitlineLws(dfi) # this adds speed-shifted columns
                except:
                    fitline = pd.DataFrame({})
                    
                try:
                    mafScore = fitline[fitline.hr == 140.0].avgSpeed.iloc[0]
                    print "MAF "
                    print mafScore
                except:
                    mafScore = np.nan
                    
                fitline_json = fitline.to_json(orient='records')
                
                 # getting summary info for run (as one-entry dict)
                runSummary = thresher.getSingleSummaryDf(dfi)
                
                # adding mafScore to summary
                runSummary['mafScore'] = mafScore
                
                print runSummary
                
                # adding predicted hr and speed values
                #dfi = thresher.getPred(dfi)

                # saving entry to database
                Activity.objects.create(act_id = activityId, name=str(activities[i].name), description=activities[i].description, act_type=activities[i].type, date=activities[i].start_date_local, timezone=activities[i].timezone, df=dfi.to_json(orient='records'), avgHr=runSummary['avgHr'], hrVar=runSummary['variation'], realMiles=runSummary['realMiles'], recovery=runSummary['recovery'], easy=runSummary['easy'], stamina=runSummary['stamina'], impulse=runSummary['impulse'], totalTime=runSummary['totalTime'], totalDist=runSummary['totalDist'], climb=runSummary['climb'], fitline=fitline_json, mafScore=mafScore, athlete=ath)
                
                # updating runs summary
                existingSummary = existingSummary.append(runSummary, ignore_index=True)
                existingFitlines = existingFitlines.append(fitline, ignore_index=True)
                masterList = masterList.append(dfi, ignore_index=True)
                
            except:
                continue    
    
    
    # saving updated runs summary to athlete profile
    ath.runsSummary = existingSummary.to_json(orient='records')
    ath.save(update_fields=['runsSummary'])
    
    existingSummary.to_pickle("runsSummary.txt")
    
    # saving updated runs summary to athlete profile
    ath.fitLines = existingFitlines.to_json(orient='records')
    ath.save(update_fields=['fitLines'])
    
    ath.masterList = masterList.to_json(orient='records')
    ath.save(update_fields=['masterList'])
    
    # testing...
    existingSummary = pd.read_json(ath.runsSummary)
    #print(existingSummary)
    
    existingFitlines = pd.read_json(ath.fitLines)
    #print(existingFitlines)

    
    global path
    path = os.path.dirname(__file__)
    # updating dataframe, pickling for use in other views
    #global df
    #df = thresher.masterAssemble(client) 
    
    masterDf = pd.read_json(ath.masterList)
    #print(masterDf)
    masterDf.to_pickle(str(path)+"/"+str(athlete.id)+"masterDf.txt")

    return render(request, 'stravaChimp/authorization.html', {'code':code, 'access_token':access_token, 'athleteId':athleteId})
 
 
# main dashboard view   
def dashboard(request, athleteId):

    print('getting athlete id...')
    athleteId = int(athleteId)
  
    #df = pd.read_pickle(str(path)+"/master_dfs/"+str(athleteId)+"masterDf.txt")
    #summaryDf = thresher.getSummaryDf(df)
    #summaryDf_json = summaryDf.to_json(orient="records")
    
    print('getting athlete object...')
    ath = Athlete.objects.get(athleteId=athleteId)
    
    print('getting run summary from athlete object...')
    summaryDf = pd.read_json(ath.runsSummary)
    
    # setting index again for use in rolling
    summaryDf = summaryDf.set_index(summaryDf.date)

    #print(summaryDf)
    summaryDf_json = ath.runsSummary
    
    # making weekly summary
    weekSummarySum = summaryDf[['climb', 'easy', 'stamina', 'recovery', 'impulse', 'realMiles', 'totalDist', 'totalTime']].resample('W', how='sum')   
    # fix variation. this isn't a helpful number
    weekSummaryMean = summaryDf[['avgHr', 'variation']].resample('W', how='mean')
    weekSummary = pd.concat([weekSummarySum, weekSummaryMean], axis=1)
    #ws_json = weekSummary.to_json(orient='records')

    #print('weekSummary')
    #print(weekSummary)
    
    #### Summary by zone WEEKLY
    recovery = DataFrame({'x':weekSummary.index, 'y':weekSummary.recovery})
    recovery = recovery.set_index(np.arange(len(recovery)))
    recoveryW_json = recovery.to_json(orient="index") # "index" used for stacked specific layout. 
    
    easy = DataFrame({'x':weekSummary.index, 'y':weekSummary.easy})
    easy = easy.set_index(np.arange(len(easy)))
    easyW_json = easy.to_json(orient="index")
    
    stamina = DataFrame({'x':weekSummary.index, 'y':weekSummary.stamina})
    stamina = stamina.set_index(np.arange(len(stamina)))
    staminaW_json = stamina.to_json(orient="index")
    
    impulse = DataFrame({'x':weekSummary.index, 'y':weekSummary.impulse})
    impulse = impulse.set_index(np.arange(len(impulse)))
    impulseW_json = impulse.to_json(orient="index")
    
    
    #### Summary by zone
    print('recovery')
    recovery = DataFrame({'x':summaryDf.date, 'y':summaryDf.recovery})
    recovery = recovery.set_index(np.arange(len(recovery)))
    recovery_json = recovery.to_json(orient="index") # "index" used for stacked specific layout. 
    print recovery_json
    print('easy')
    easy = DataFrame({'x':summaryDf.date, 'y':summaryDf.easy})
    easy = easy.set_index(np.arange(len(easy)))
    easy_json = easy.to_json(orient="index")
    
    stamina = DataFrame({'x':summaryDf.date, 'y':summaryDf.stamina})
    stamina = stamina.set_index(np.arange(len(stamina)))
    stamina_json = stamina.to_json(orient="index")
    
    impulse = DataFrame({'x':summaryDf.date, 'y':summaryDf.impulse})
    impulse = impulse.set_index(np.arange(len(impulse)))
    impulse_json = impulse.to_json(orient="index")
    
    rList = list(recovery)
    eList = list(easy)
    sList = list(stamina)
    iList = list(impulse)
    stackedTester = [rList, eList, sList, iList]
    print stackedTester




    print('rolling')
    ### rolling
    rollDf = thresher.getRollingSummaryDf(summaryDf)
    rollDf =rollDf.fillna(0)
    rollingDf_json = rollDf.to_json(orient="records")

    rollDfLastFive_json = rollDf.sort_values(['date'], ascending=False)[:5].to_json(orient="records")




    print('rollrec')
    rollRec = DataFrame({'x':rollDf.date, 'y':rollDf.rollRec})
    rollRec["y"] = rollRec.y + 100
    rollRec = rollRec.set_index(np.arange(len(rollRec)))
    rollRec_json = rollRec.to_json(orient="index")

    print('rolleasy')
    rollEasy = DataFrame({'x':rollDf.date, 'y':rollDf.rollEasy})
    rollEasy = rollEasy.set_index(np.arange(len(rollEasy)))
    rollEasy_json = rollEasy.to_json(orient="index")

    print('rollstam')
    rollStam = DataFrame({'x':rollDf.date, 'y':rollDf.rollStam})
    rollStam = rollStam.set_index(np.arange(len(rollStam)))
    rollStam_json = rollStam.to_json(orient="index")

    print('rollimp')
    rollImp = DataFrame({'x':rollDf.date, 'y':rollDf.rollImp})
    rollImp = rollImp.set_index(np.arange(len(rollImp)))
    rollImp_json = rollImp.to_json(orient="index")
    
    # dealing w fitlines
    #testLine = Activity.objects.get(act_id=535167099).fitline
    
    aWeekAgo = datetime.datetime.now() - datetime.timedelta(days=7)
    last7Days = summaryDf[summaryDf.date > aWeekAgo]
    dist7 = round(np.sum(last7Days.totalDist), 1)
    try:
        var7 = int(round(np.mean(last7Days.variation), 0)) # weight this average
        avgHr7 = int(round(np.mean(last7Days.avgHr), 0)) # weight this average
    except:
        var7 = 0.0
        avgHr7 = 0.0
    climb7 = int(round(np.sum(last7Days.climb), 0))
    time7 = round(np.sum(last7Days.totalTime), 1)
    trainingEfficiency7 = (dist7 - np.sum(last7Days.stamina)) / (dist7 +1)
    
    #print "7DAYs"
    #print last7Days
    #print time7
    



    ################################################
    # for maf pts

    mafPts = summaryDf[['date','mafScore']].dropna().sort_values(['date'], ascending=False)
    mafPts_json = mafPts.to_json(orient="records")
    
    mafPts_smoothed = thresher.makeLws(df=mafPts, frac=.20)
    mafPts_smoothed['date'] = mafPts_smoothed.date / 1000000 #lws messed up the dates
    mafPts_smoothed_json = mafPts_smoothed.to_json(orient="records")
    #print mafPts_smoothed_json
    
    print "\n\n"
    print "mafPts"
    print mafPts
    
    # for maf dash
    lastFive = mafPts[:5]
    lastFive_json = lastFive.to_json(orient="records")
    ath.mafLastFive = lastFive_json
    ath.save(update_fields=['mafLastFive'])

    print "lastfive"
    print lastFive
    
    # current maf score
    currentMaf = str(np.round(lastFive.mafScore.iloc[0], 2))

    athName = ath.name

    print('rendering...')

    print summaryDf



    """
    taking these out of render package:
    'tE7':trainingEfficiency7, 'dist7':dist7, 'var7':var7, 'avgHr7':avgHr7, 'climb7':climb7, 'time7':time7,
    'rollingDf_json':rollingDf_json, 'rollRec_json':rollRec_json, 'rollEasy_json':rollEasy_json, 'rollStam_json':rollStam_json, 'rollImp_json':rollImp_json,
    'fitlinesAll':ath.fitLines
    """
    return render(request, 'stravaChimp/dashboard.html', {'athName':athName, 'stackedTester':stackedTester, 'r5':rollDfLastFive_json, 'summaryDf_json':summaryDf_json, 'recovery_json':recovery_json, 'easy_json':easy_json, 'stamina_json':stamina_json, 'impulse_json':impulse_json,'recoveryW_json':recoveryW_json, 'easyW_json':easyW_json, 'staminaW_json':staminaW_json, 'impulseW_json':impulseW_json, 'athleteId':athleteId, 'ath':ath, 'mafPts_json':mafPts_json, 'currentMaf': currentMaf, 'maf_json':lastFive_json, 'mafPts_smoothed_json':  mafPts_smoothed_json, 'datesList':summaryDf['date'].to_json(orient='records')})

def run_detail(request, athleteId, activityId):

    activity = Activity.objects.get(act_id=activityId)
    ath = Athlete.objects.get(athleteId=athleteId)
    athName = ath.name

    act_date = activity.date
    name_string = str(activity.name)
    #df = pd.read_pickle(str(path)+"/master_dfs/"+str(athleteId)+"masterDf.txt")
    df = pd.read_json(activity.df)
    r = df[df.activityId == int(activityId)].loc[:,['latlng', 'hr', 'speeds','speedsSmoothed2', 'speedDelta3Abs', 'altDeltas', 'time', 'distCum', 'cadence']]
    #r = thresher.basicClean(r)
    r_json = r.to_json(orient="records")
    print r
    fitline = thresher.getFitLine(df)
    fitline_json = fitline.to_json(orient='records')
    
    return render(request, 'stravaChimp/run_detail.html', {'athName':athName, 'activity': activity, 'r_json':r_json, 'athleteId':athleteId, 'ath':ath, 'fitline_json':fitline_json, 'act_date':act_date, 'name_string':name_string})

