# Premier Leage Predictor: predicts results for the next 'future_games' games based on
# a model of Prof David Spiegelhalter

import pandas as pd
import datetime
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# List of arguments for the model:
# today = datetime.date object, date to work from, between 19/08/2016 and today (before last day of a season, and after the first)
# w = float object, weighting for home vs away between 0 and 1
# future_games = int object, number of games to predict (at most the number of games left between today and end of season)


def premGetSeason(today = datetime.date.today()):
    seasons = [1617, 1718, 1819, 1920]
    if (today.year == 2016) | ((today.year == 2017) & (today.month<8)):
        season = seasons[0]
    elif ((today.year == 2017) & (today.month >= 8)) | ((today.year == 2018) & (today.month<8)):
        season = seasons[1]
    elif ((today.year == 2018) & (today.month >= 8)) | ((today.year == 2019) & (today.month<8)):
        season = seasons[2]
    elif ((today.year == 2019) & (today.month >= 8)) | (today.year == 2020):
        season = seasons[3]

    return season




def premGetScores(today = datetime.date.today()):# Obtain file containing scores as csv
    season = premGetSeason(today)
    site = "http://www.football-data.co.uk/"
    filename = site + "mmz4281/"+ str(season) + "/E0.csv"
    prem = pd.read_csv(filename)
    return prem




def premLeagueStrengths(today = datetime.date.today(), w = 0.33): # Create strengths df
    
    prem = premGetScores(today)
    
    hometeam = prem.loc[:, "HomeTeam"].value_counts().sort_index()
    awayteam = prem.loc[:, "AwayTeam"].value_counts().sort_index()
    
    homegoals = prem.loc[:, ["HomeTeam", "FTHG"]].groupby(["HomeTeam"]).agg({'FTHG':sum})
    awaygoals = prem.loc[:, ["AwayTeam", "FTAG"]].groupby(["AwayTeam"]).agg({'FTAG':sum})
    
    homeconc = prem.loc[:, ["HomeTeam", "FTAG"]].groupby(["HomeTeam"]).agg({'FTAG':sum})
    awayconc = prem.loc[:, ["AwayTeam", "FTHG"]].groupby(["AwayTeam"]).agg({'FTHG':sum})
    
    strengths = pd.concat([hometeam, awayteam, homegoals, awaygoals, homeconc, awayconc], axis = 1)
    strengths.columns = ['HomeGames', 'AwayGames', 'HomeGoals', 'AwayGoals', 'HomeConcede', 'AwayConcede']
    
    strengths['AvgHomeGoals'] = strengths.apply(lambda x: x.HomeGoals/x.HomeGames, axis=1)
    strengths['AvgAwayGoals'] = strengths.apply(lambda x: x.AwayGoals/x.AwayGames, axis=1)
    
    meanHomeGoals = sum(strengths['HomeGoals'])/sum(strengths['HomeGames'])
    meanAwayGoals = sum(strengths['AwayGoals'])/sum(strengths['AwayGames'])
    
    strengths['WtdHomeAttackStrength'] = strengths.apply(lambda x: ((1-w)*x.AvgHomeGoals + w*x.AvgAwayGoals)/((1-w)*meanHomeGoals + w*meanAwayGoals), axis = 1)
    strengths['WtdAwayAttackStrength'] = strengths.apply(lambda x: (w*x.AvgHomeGoals + (1-w)*x.AvgAwayGoals)/(w*meanHomeGoals + (1-w)*meanAwayGoals), axis = 1)
    
    strengths['AvgHomeConcede'] = strengths.apply(lambda x: x.HomeConcede/x.HomeGames, axis=1)
    strengths['AvgAwayConcede'] = strengths.apply(lambda x: x.AwayConcede/x.AwayGames, axis=1)
    
    meanHomeConcede = sum(strengths['HomeConcede'])/sum(strengths['HomeGames'])
    meanAwayConcede = sum(strengths['AwayConcede'])/sum(strengths['AwayGames'])
    
    strengths['WtdHomeDefenceWeakness'] = strengths.apply(lambda x: ((1-w)*x.AvgHomeConcede + w*x.AvgAwayConcede)/((1-w)*meanHomeConcede + w*meanAwayConcede), axis = 1)
    strengths['WtdAwayDefenceWeakness'] = strengths.apply(lambda x: (w*x.AvgHomeConcede + (1-w)*x.AvgAwayConcede)/(w*meanHomeConcede + (1-w)*meanAwayConcede), axis = 1)

    return strengths





def premMeanHomeGoals(today = datetime.date.today(), w = 0.33):
    strengths = premLeagueStrengths(today, w)
    meanHomeGoals = sum(strengths['HomeGoals'])/sum(strengths['HomeGames'])
    return meanHomeGoals



def premMeanAwayGoals(today = datetime.date.today(), w = 0.33):
    strengths = premLeagueStrengths(today, w)
    meanAwayGoals = sum(strengths['AwayGoals'])/sum(strengths['AwayGames'])
    return meanAwayGoals




def premGetFixtures(today = datetime.date.today()):    # Obtain file containing future fixtures as csv
    
    strengths = premLeagueStrengths(today, 0.33)
    
    def seasonYear(today):
        if today.month < 8:
            return today.year - 1
        else:
            return today.year
    
    filename2 = 'https://fixturedownload.com/download/epl-' + str(seasonYear(today)) +'-GMTStandardTime.csv' 
    fixtures = pd.read_csv(filename2) 
    fixtures['Date'] = pd.to_datetime(fixtures['Date'], format = '%d/%m/%Y %H:%M')
    
    # Filter for future dates, saving Date and both Teams only
    fixtures = fixtures[fixtures['Date'] >= pd.Timestamp(today)].filter(['Date', 'Home Team', 'Away Team']).set_index('Date')
    
    # Match team names
    teams1 = strengths.index
    teams2 = sorted(list(dict.fromkeys(list(fixtures['Home Team'])+list(fixtures['Away Team']))))
    teams_dictionary = {teams2[i]:teams1[i] for i in range(20)}
    
    fixtures['Home Team'] = fixtures.apply(lambda x: teams_dictionary[x['Home Team']], axis = 1)
    fixtures['Away Team'] = fixtures.apply(lambda x: teams_dictionary[x['Away Team']], axis = 1)
    
    return fixtures




    
def premLeaguePredictions(today = datetime.date.today(), w = 0.33, future_games = 10):    # Create predictions df
    
    fixtures = premGetFixtures(today)
    strengths = premLeagueStrengths(today, w)
    
    predictions = fixtures.copy().iloc[:future_games]
    predictions.columns = ['HomeTeam', 'AwayTeam']
    
    predictions['AvgHomeGoals'] = premMeanHomeGoals()
    
    predictions['WtdHomeAttackStrength'] = predictions.apply(lambda x: strengths.loc[x['HomeTeam'], 'WtdHomeAttackStrength'], axis = 1)
    
    predictions['WtdAwayDefenceWeakness'] = predictions.apply(lambda x: strengths.loc[x['AwayTeam'], 'WtdAwayDefenceWeakness'], axis = 1)
    
    predictions['ExpectedHomeGoals'] = predictions.apply(lambda x: x['AvgHomeGoals'] * x['WtdHomeAttackStrength'] * x['WtdAwayDefenceWeakness'], axis = 1)
    
    predictions['AvgAwayGoals'] = premMeanAwayGoals()
    
    predictions['WtdAwayAttackStrength'] = predictions.apply(lambda x: strengths.loc[x['AwayTeam'], 'WtdAwayAttackStrength'], axis = 1)
    
    predictions['WtdHomeDefenceWeakness'] = predictions.apply(lambda x: strengths.loc[x['HomeTeam'], 'WtdHomeDefenceWeakness'], axis = 1)
    
    predictions['ExpectedAwayGoals'] = predictions.apply(lambda x: x['AvgAwayGoals'] * x['WtdHomeDefenceWeakness'] * x['WtdAwayAttackStrength'], axis = 1)


    # Calculate probabilities
    possible_number_goals = [0, 1, 2, 3, 4]
    home_or_away = ['Home', 'Away']
    
    for i in home_or_away:
        for j in possible_number_goals:
            predictions['P(' + i + 'Goals = ' + str(j) + ')' ] = predictions.apply(lambda x: poisson.pmf(j, x['Expected' + i + 'Goals']), axis = 1)
    
        predictions['P(' + i + 'Goals = 5+)'] = predictions.apply(lambda x: 1 - poisson.cdf(4, x['Expected' + i + 'Goals']), axis = 1) 
    
    predictions['P(HomeWin)'] = predictions.apply(lambda x: x['P(HomeGoals = 1)']*x['P(AwayGoals = 0)'] + x['P(HomeGoals = 2)']*(x['P(AwayGoals = 0)'] + x['P(AwayGoals = 1)']) \
               + x['P(HomeGoals = 3)']*(x['P(AwayGoals = 0)'] + x['P(AwayGoals = 1)'] + x['P(AwayGoals = 2)']) + x['P(HomeGoals = 4)']*(x['P(AwayGoals = 0)'] + x['P(AwayGoals = 1)'] + x['P(AwayGoals = 2)'] + x['P(AwayGoals = 3)']) \
               + x['P(HomeGoals = 5+)']*(x['P(AwayGoals = 0)'] + x['P(AwayGoals = 1)'] + x['P(AwayGoals = 2)'] + x['P(AwayGoals = 3)'] + x['P(AwayGoals = 4)']), axis = 1)
    predictions['P(Draw)'] = predictions.apply(lambda x: x['P(HomeGoals = 0)']*x['P(AwayGoals = 0)'] + x['P(HomeGoals = 1)']*x['P(AwayGoals = 1)'] + x['P(HomeGoals = 2)']*x['P(AwayGoals = 2)'] \
               + x['P(HomeGoals = 3)']*x['P(AwayGoals = 3)'] + x['P(HomeGoals = 4)']*x['P(AwayGoals = 4)'] + x['P(HomeGoals = 5+)']*x['P(AwayGoals = 5+)'], axis = 1)
    predictions['P(AwayWin)'] = predictions.apply(lambda x: 1 - (x['P(HomeWin)'] + x['P(Draw)']), axis = 1)

    return predictions








def premLeagueGraphs(today = datetime.date.today(), w = 0.33, future_games = 10):    # Display results
    predictions = premLeaguePredictions(today, w, future_games)
    
    labels1 = ['Home win', 'Draw', 'Away win']
    probabilities1 = predictions.loc[:, ['P(HomeWin)', 'P(Draw)', 'P(AwayWin)']]
    
    _x = np.array([0, 1, 2, 3, 4, 5]) # away goals
    _y = np.array([0, 1, 2, 3, 4, 5]) # home goals
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()   # list of possible scores, where y = home goals, x = away goals
    
    
    for i in range(future_games):
        plt.pie(list(probabilities1.iloc[i]), labels=labels1, shadow = True, autopct='%1.1f%%') # pie charts
        plt.title(predictions.iloc[i, 0] + ' vs ' + predictions.iloc[i, 1])
        plt.show()
    
    
        plt.clf()  
    
    
        fig = plt.figure(figsize=(18, 8))   # 3D bar plots
        ax1 = fig.add_subplot(121, projection='3d')
        
        z = []
        for k in _x:
            prob = []
            for j in _y:
                prob.append(predictions.iloc[i, 10+k]*predictions.iloc[i, 16+j])
            z.append(prob)
        _z = np.array(z) # creates list of probabilities with z[i][j] = P(away = x[i], home = y[j])
    
        z = _z.ravel() 
        
        top = z
        bottom = np.zeros_like(top)
        width = depth = 0.5
        
        ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
        ax1.set_title(predictions.iloc[i, 0] + ' vs ' + predictions.iloc[i, 1])
        ax1.set_xlabel('Away goals')
        ax1.set_ylabel('Home goals')
        ax1.set_zlabel('Probability')
        plt.xticks(x, [0, 1, 2, 3, 4, '5+'])
        plt.yticks(x, [0, 1, 2, 3, 4, '5+'])
        
        
        plt.show()
        
        plt.clf()
    
    
def premLeagueSummary(today = datetime.date.today(), w = 0.33, future_games = 10): # Create summary df
    predictions = premLeaguePredictions(today, w, future_games)
    
    summary = predictions.iloc[:, 0:2]
    
    def obtainMax(df_row):
        if max(df_row['P(HomeWin)'],df_row['P(Draw)'], df_row['P(AwayWin)']) == df_row['P(HomeWin)']:
            return 'Home Win'
        elif max(df_row['P(HomeWin)'],df_row['P(Draw)'], df_row['P(AwayWin)']) == df_row['P(Draw)']:
            return 'Draw'
        else:
            return 'Away Win'
        
    def obtainScore(df_row):
        MaxHome = 0
        MaxHomeIndex = 0
        MaxAway = 0
        MaxAwayIndex = 0
        for i, j in enumerate(df_row[10:16]):
            if j > MaxHome:
                MaxHome = j
                MaxHomeIndex = i
        for i, j in enumerate(df_row[16:22]):
            if j > MaxAway:
                MaxAway = j
                MaxAwayIndex = i
        if (MaxHomeIndex == 5) & (MaxAwayIndex == 5):
            return ('5+', MaxHome, '5+', MaxAway)
        elif (MaxHomeIndex == 5) & (MaxAwayIndex != 5):
            return ('5+', MaxHome, MaxAwayIndex, MaxAway)
        elif (MaxHomeIndex != 5) & (MaxAwayIndex == 5):
            return (MaxHomeIndex, MaxHome, '5+', MaxAway)
        else:
            return (MaxHomeIndex, MaxHome, MaxAwayIndex, MaxAway)
    
        
    summary['LikelyResult'] = predictions.apply(obtainMax, axis = 1)
    
    summary['P(LikelyResult)'] = predictions.apply(lambda x: max(x['P(HomeWin)'], x['P(Draw)'], x['P(AwayWin)']), axis = 1)
    
    summary['LikelyScore'] = predictions.apply(lambda x: (obtainScore(x)[0], obtainScore(x)[2]), axis = 1)
    
    summary['P(LikelyScore)'] = predictions.apply(lambda x: obtainScore(x)[1]*obtainScore(x)[3], axis = 1)
    
    return summary

        
