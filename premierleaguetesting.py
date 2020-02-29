# Premier Leage Predictor: backtests the model using historical data. 

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import premierleaguepredictor as plp


# List of arguments:
# predicted_games = int object, number of predicted games to compare
# today = datetime.date object, date to work back from (at least a week past predicted_games into any season from 2016/2017)
# w = float object, usual weighting of home vs away




def premLeagueResults(today = datetime.date.today(), predicted_games = 10):
    results = plp.premGetScores(today)
    results['Date'] = pd.to_datetime(results['Date'], format = '%d/%m/%Y')
    results = results[results['Date'] <= pd.Timestamp(today)].filter(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']).set_index('Date')
    results = results.iloc[-predicted_games:, :]
    results.columns = ['HomeTeam', 'AwayTeam','HomeGoals', 'AwayGoals']
    
    return results
    




def premLeagueCompare(today = datetime.date.today(), predicted_games = 10, w = 0.33):
    results = premLeagueResults(today, predicted_games)
    predictions = plp.premLeagueSummary(today, w, predicted_games)
    predictions.index = results.index
    
    def getResult(df_row):
        if df_row['HomeGoals'] > df_row['AwayGoals']:
            return 'Home Win'
        elif df_row['HomeGoals'] < df_row['AwayGoals']:
            return 'Away Win'
        else:
            return 'Draw'
    
    results['Result'] = results.apply(getResult, axis = 1)
    
    results['PHomeGoals'] = predictions.apply(lambda x: x['LikelyScore'][0], axis = 1)
    results['PAwayGoals'] = predictions.apply(lambda x: x['LikelyScore'][1], axis = 1)
    
    results['PResult'] = predictions['LikelyResult']
    
    return results




def premLeagueCompareSummary(today = datetime.date.today(), predicted_games = 10, w = 0.33):
    prediction_score = {}
    result_count = 0
    score_count = 0
    compare = premLeagueCompare(today, predicted_games, w)
    for i in range(predicted_games):
        if compare.iloc[i, :]['Result'] == compare.iloc[i, :]['PResult']:
            result_count += 1
        if (compare.iloc[i, :]['HomeGoals'] == compare.iloc[i, :]['PHomeGoals']) & \
        (compare.iloc[i, :]['AwayGoals'] == compare.iloc[i, :]['PAwayGoals']):
            score_count += 1
    prediction_score['results'] = result_count
    prediction_score['scores'] = score_count
    prediction_score['total'] = result_count + 4*score_count
    
    return prediction_score
    




def premLeagueWeighting(today = datetime.date.today(), predicted_games = 10): # find w with highest score, currently extremely inefficient
    results = []
    scores = []
    totals = []
    w_range = [i*0.05 for i in range(0, 21)]
    
    for w in w_range:
        summary = premLeagueCompareSummary(today, predicted_games, w)
        results.append(summary['results'])
        scores.append(summary['scores'])
        totals.append(summary['total'])
    
    plt.plot(w_range, results)
    plt.plot(w_range, scores)
    plt.plot(w_range, totals)
    
    plt.show()
        
        
        


    
    

