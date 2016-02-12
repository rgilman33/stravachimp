from django.shortcuts import render
import numpy as np
from urllib2 import urlopen, Request
from json import load, dumps
import requests
from django.http import HttpRequest 
import pickle, pandas
from pandas import DataFrame, Series
from stravalib import Client
#import urllib3.contrib.pyopenssl
#urllib3.contrib.pyopenssl.inject_into_urllib3()
import thresher

client = Client()

# Got from Strava after registering on website
MY_STRAVA_CLIENT_ID = "9558"
MY_STRAVA_CLIENT_SECRET = "734e394ff653703abaef7c7c061cc8d685241a10"

# a url sending visitor to strava's website for authentication
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
    global client
    client = Client(access_token=access_token)
    athlete = client.get_athlete() # Get current athlete details
    
    # saving access_token as pickle for use in other views    
    outputFile = open("stravaChimp/tokens/"+str(athlete.id)+"token.txt", 'w')
    pickle.dump(access_token, outputFile)
    outputFile.close()   
    
    # updating dataframe, pickling for use in other views
    global df
    df = thresher.masterAssemble(client) 
    oFile = open("stravaChimp/master_dfs/"+str(athlete.id)+"masterDf.txt", 'w')
    pickle.dump(df, oFile)
    oFile.close() 
    
    return render(request, 'stravaChimp/authorization.html', {'code':code, 'access_token':access_token,})
 
 
# main dashboard view   
def dashboard(request):
    """
    athlete = client.get_athlete()
    
    iFile = open("stravaChimp/master_dfs/"+str(athlete.id)+"masterDf.txt", 'r')
    df = pickle.load(iFile)
    iFile.close()
    """
    summaryDf = thresher.getSummaryDf(df)
    summaryDf_json = summaryDf.to_json(orient="records")
    rollingDf = thresher.getRollingSummaryDf(summaryDf)
    rollingDf_json = rollingDf.to_json(orient="records")
    
    #### Summary by zone
    recovery = DataFrame({'x':summaryDf.date, 'y':summaryDf.recovery})
    recovery = recovery.set_index(np.arange(len(recovery)))
    recovery_json = recovery.to_json(orient="index") # "index" used for stacked specific layout. 
    
    easy = DataFrame({'x':summaryDf.date, 'y':summaryDf.easy})
    easy = easy.set_index(np.arange(len(easy)))
    easy_json = easy.to_json(orient="index")
    
    stamina = DataFrame({'x':summaryDf.date, 'y':summaryDf.stamina})
    stamina = stamina.set_index(np.arange(len(stamina)))
    stamina_json = stamina.to_json(orient="index")
    
    impulse = DataFrame({'x':summaryDf.date, 'y':summaryDf.impulse})
    impulse = impulse.set_index(np.arange(len(impulse)))
    impulse_json = impulse.to_json(orient="index")
    
    ### rolling
    rollDf = thresher.getRollingSummaryDf(summaryDf)
    rollDf =rollDf.fillna(0)

    rollRec = DataFrame({'x':rollDf.date, 'y':rollDf.rollRec})
    rollRec["y"] = rollRec.y + 100
    rollRec = rollRec.set_index(np.arange(len(rollRec)))
    rollRec_json = rollRec.to_json(orient="index")

    rollEasy = DataFrame({'x':rollDf.date, 'y':rollDf.rollEasy})
    rollEasy = rollEasy.set_index(np.arange(len(rollEasy)))
    rollEasy_json = rollEasy.to_json(orient="index")

    rollStam = DataFrame({'x':rollDf.date, 'y':rollDf.rollStam})
    rollStam = rollStam.set_index(np.arange(len(rollStam)))
    rollStam_json = rollStam.to_json(orient="index")

    rollImp = DataFrame({'x':rollDf.date, 'y':rollDf.rollImp})
    rollImp = rollImp.set_index(np.arange(len(rollImp)))
    rollImp_json = rollImp.to_json(orient="index")

    return render(request, 'stravaChimp/dashboard.html', {'summaryDf_json':summaryDf_json, 'rollingDf_json':rollingDf_json, 'recovery_json':recovery_json, 'easy_json':easy_json, 'stamina_json':stamina_json, 'impulse_json':impulse_json, 'rollRec_json':rollRec_json, 'rollEasy_json':rollEasy_json, 'rollStam_json':rollStam_json, 'rollImp_json':rollImp_json})

def run_detail(request, pk):

    r = df[df.activityId == int(pk)].loc[:,['latlng', 'hr', 'speeds', 'speedDelta3Abs', 'altDeltas', 'time', 'distCum']]
    r_json = r.to_json(orient="records")
    
    return render(request, 'stravaChimp/run_detail.html', {'r_json':r_json})

