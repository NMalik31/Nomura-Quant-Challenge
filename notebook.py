#NOMURA QUANT CHALLENGE 2025

#The format for the weights dataframe for the backtester is attached with the question.
#Complete the below codes wherever applicable

import pandas as pd
import numpy as np
import os

# def backtester_without_TC(weights_df):
#     #Update data file path here
#     data = pd.read_csv('cross_val_data.csv')

#     weights_df = weights_df.fillna(0)

#     start_date = 3500
#     end_date = 3999

#     initial_notional = 1

#     df_returns = pd.DataFrame()

#     for i in range(0,20):
#         data_symbol = data[data['Symbol']==i]
#         data_symbol = data_symbol['Close']
#         data_symbol = data_symbol.reset_index(drop=True)   
#         data_symbol = data_symbol/data_symbol.shift(1) - 1
#         df_returns =  pd.concat([df_returns,data_symbol], axis=1, ignore_index=True)
    
#     df_returns = df_returns.fillna(0)
    
#     weights_df = weights_df.loc[start_date:end_date]    
#     df_returns = df_returns.loc[start_date:end_date]

#     df_returns = weights_df.mul(df_returns)
#     print(df_returns.head(10))
#     print(df_returns.isnull().sum().sum())
#     print(df_returns.sum().sum())

#     notional = initial_notional

#     returns = []

#     for date in range(start_date,end_date+1):
#         returns.append(df_returns.loc[date].values.sum())
#         notional = notional * (1+returns[date-start_date])

#     net_return = ((notional - initial_notional)/initial_notional)*100
#     sharpe_ratio = (pd.DataFrame(returns).mean().values[0])/pd.DataFrame(returns).std().values[0]

#     return [net_return, sharpe_ratio]

def backtester_without_TC(weights_df):
    import pandas as pd
    import numpy as np

    data = pd.read_csv('cross_val_data.csv')

    weights_df = weights_df.fillna(0)

    start_date = 3500
    end_date = 3999

    initial_notional = 1

    # Build the returns DataFrame properly indexed by date
    df_returns = pd.DataFrame(index=range(start_date, end_date + 1), columns=range(20), dtype=float)

    for i in range(20):
        data_symbol = data[data['Symbol'] == i].sort_values('Date')
        data_symbol = data_symbol.set_index('Date')
        # Compute returns for each date
        symbol_returns = data_symbol['Close'].pct_change().fillna(0)
        # Restrict to the backtest period
        symbol_returns = symbol_returns.loc[start_date:end_date]
        # Assign to DataFrame
        df_returns[i] = symbol_returns

    df_returns = df_returns.fillna(0)

    # At this point, df_returns and weights_df have the same index (dates)
    df_returns = weights_df.mul(df_returns)

    notional = initial_notional
    returns = []

    for idx, date in enumerate(range(start_date, end_date + 1)):
        daily_return = df_returns.loc[date].values.sum()
        returns.append(daily_return)
        notional = notional * (1 + daily_return)

    returns = np.array(returns)
    net_return = ((notional - initial_notional) / initial_notional) * 100
    sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0

    return [net_return, sharpe_ratio]


def task1_Strategy1():
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('cross_val_data.csv')
    output_df = pd.DataFrame(0.0, index = range(3500, 4000), columns = range(20)) 

    #Dictionary of stock price time series for crossval_data
    close_dict = {}
    for symbol in range(20):
        close_dict[symbol] = crossval_data[crossval_data['Symbol'] == symbol].set_index('Date')['Close']
    #Given days 3500 to 3999, and 1 week is 5 days, so we are using range(0, 100) to be safe
    for week in range(0, 100):
        week_start = 3500 + week * 5
        week_end = week_start + 4
        if week_end > 3999:
            break
        #Now we will compute mean returns for each stock for latest 50 completed weeks
        mean_returns = []
        for symbol in range(20):
            #Full closing price for given symbol
            full_close = pd.concat([
                train_data[train_data['Symbol'] == symbol].set_index('Date')['Close'],
                crossval_data[crossval_data['Symbol'] == symbol].set_index('Date')['Close']
            ])
            prev_week_end = week_start - 1
            week_returns = []
            for w in range(50):
                this_week_end = prev_week_end - w * 5
                this_week_start = this_week_end - 4
                if this_week_start < 0:
                    break
                last_close = full_close.get(this_week_end, np.nan)
                prev_last_close = full_close.get(this_week_start - 1, 1.0 if this_week_start - 1 < 0 else np.nan)
                if np.isnan(last_close) or np.isnan(prev_last_close) or prev_last_close == 0:
                    continue
                week_return = (last_close - prev_last_close) / prev_last_close
                week_returns.append(week_return)
            if len(week_returns) > 0:
                mean_return = np.mean(week_returns)
            else:
                mean_return = 0
            mean_returns.append((symbol, mean_return))
        #Now we will rank stocks by mean returns in decreasing order
        mean_returns.sort(key = lambda x: -x[1])
        top6 = [x[0] for x in mean_returns[:6]] #Dates corresponding to the best 6 stocks by mean weekly returns
        bottom6 = [x[0] for x in mean_returns[-6:]] #Dates corresponding to the worst 6 stocks by mean weekly returns
        #Assign weights of -1/6 to best 6 stocks, and +1/6 to worst 6 stocks, and 0 to the rest
        for symbol in top6:
            output_df.loc[range(week_start, week_end + 1), symbol] = -1/6
        for symbol in bottom6:
            output_df.loc[range(week_start, week_end + 1), symbol] = 1/6
    return output_df


def task1_Strategy2():
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('cross_val_data.csv')
    output_df = pd.DataFrame(0.0, index = range(3500, 4000), columns = range(20))

    data = pd.concat([train_data, crossval_data], ignore_index = True)
    #Run loop for each date from 3500 to 3999 in crossval data
    for date in range(3500, 4000):
        relative_positions = []
        for symbol in range(20):
            symbol_data = data[data['Symbol'] == symbol].set_index('Date')['Close']
            #Now we get last 30 and last 5 closes for LMA and SMA calculations
            closes_30 = symbol_data.loc[date - 30 : date - 1] if (date - 30 >= 0) else symbol_data.loc[:date - 1]
            closes_5 = symbol_data.loc[date - 5 : date - 1] if (date - 5 >= 0) else symbol_data.loc[:date - 1]
            if len(closes_30) < 30 or len(closes_5) < 5:
                relative_positions.append((symbol, 0))
                continue
            #Calculate LMA and SMA
            LMA = closes_30.mean()
            SMA = closes_5.mean()
            if LMA == 0 or np.isnan(LMA) or np.isnan(SMA):
                relative_positions.append((symbol, 0))
                continue
            #Calculate relative position
            relative_position = (SMA - LMA) / LMA
            relative_positions.append((symbol, relative_position))
        #Sort relative positions in decreasing order
        relative_positions.sort(key = lambda x: -x[1])
        top5 = [x[0] for x in relative_positions[:5]] #Dates corresponding to the best 5 stocks according to moving averages
        bottom5 = [x[0] for x in relative_positions[-5:]] #Dates corresponding to the worst 5 stocks according to moving averages
        #Assign weights of -1/5 to best 5 stocks, and +1/5 to worst 5 stocks, and 0 to the rest
        for symbol in top5:
            output_df.loc[date, symbol] = -1/5
        for symbol in bottom5:
            output_df.loc[date, symbol] = 1/5
    
    return output_df


def task1_Strategy3():
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('cross_val_data.csv')
    output_df = pd.DataFrame(0.0, index = range(3500, 4000), columns = range(20))

    data = pd.concat([train_data, crossval_data], ignore_index = True)
    #Basic loop idea is same as for Strategy 2, only change is that now we calculate and sort by ROC instead
    for date in range(3500, 4000):
        rocs = []
        for symbol in range(20):
            symbol_data = data[data['Symbol'] == symbol].set_index('Date')['Close']
            #We check whether we are not missing any data or passing problematic data
            if (date - 7 < 0):
                rocs.append((symbol, 0))
                continue
            latest_close = symbol_data.get(date, np.nan)
            prev_close = symbol_data.get(date - 7, np.nan)
            if np.isnan(latest_close) or np.isnan(prev_close) or prev_close == 0:
                rocs.append((symbol, 0))
                continue
            roc = (latest_close - prev_close) / prev_close
            rocs.append((symbol, roc))
        #Sort rocs in decreasing order
        rocs.sort(key = lambda x: -x[1])
        top4 = [x[0] for x in rocs[:4]] #Dates corresponding to the best 4 stocks according to ROC
        bottom4 = [x[0] for x in rocs[-4:]] #Dates corresponding to the worst 4 stocks according to ROC
        #Assign weights of -1/4 to best 4 stocks, and +1/4 to worst 4 stocks, and 0 to the rest
        for symbol in top4:
            output_df.loc[date, symbol] = -1/4
        for symbol in bottom4:
            output_df.loc[date, symbol] = 1/4
    
    return output_df


def task1_Strategy4():
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('cross_val_data.csv')
    output_df = pd.DataFrame(0.0, index = range(3500, 4000), columns = range(20))

    data = pd.concat([train_data, crossval_data], ignore_index = True)
    for date in range(3500, 4000):
        proximities_support = []
        proximities_resistances = []
        latest_closes = [] #These 3 lists will help us calculate of latest closing prices with the support and resistance values 
        for symbol in range(20):
            symbol_data = data[data['Symbol'] == symbol].set_index('Date')['Close']
            #We use latest 21 closes, including the previous day 
            closes_21 = symbol_data.loc[date - 21 : date - 1] if (date - 21 >= 0) else symbol_data.loc[:date - 1]
            #Handling missing data or data that can cause problems in later calculations
            if len(closes_21) < 21:
                proximities_support.append((symbol, np.inf)) #We assign an infinity value so as to ensure that this stock is not chosen
                proximities_resistances.append((symbol, -np.inf)) #Same reasoning as above
                latest_closes.append((symbol, np.nan))
                continue
            SMA = closes_21.mean()
            std = closes_21.std()
            support = SMA - 3 * std
            resistance = SMA + 3 * std
            latest_close = symbol_data.get(date, np.nan)
            if np.isnan(latest_close) or np.isnan(SMA) or np.isnan(std) or support == 0 or resistance == 0:
                proximities_support.append((symbol, np.inf))
                proximities_resistances.append((symbol, -np.inf))
                latest_closes.append((symbol, np.nan))
                continue
            prox_to_resistance = (latest_close - resistance) / resistance
            prox_to_support = (latest_close - support) / support
            proximities_support.append((symbol, prox_to_support))
            proximities_resistances.append((symbol, prox_to_resistance))
            latest_closes.append((symbol, latest_close))
        
        #Ranking stocks based on proximity to support in increasing order
        proximity_support_sorted = sorted(proximities_support, key = lambda x: x[1])
        #We take the dates for the top 4 stocks according to proximity to support ranking
        top4_support = [x[0] for x in proximity_support_sorted[:4]]
        #Now, we will remove these from our data and then sort the remaining according to proximity to resistance in decreasing order
        support_vals = set(top4_support)
        remaining = [x for x in proximities_resistances if x[0] not in support_vals]
        remaining_sorted = sorted(remaining, key = lambda x: -x[1])
        #Finally, we take the dates for the top 4 stocks according to proximity to resistance ranking
        top4_resistance = [x[0] for x in remaining_sorted[:4]]
        #We assign weights of +1/4 to top 4 support stocks, and -1/4 to top 4 resistance stocks, and 0 to the rest
        for symbol in top4_support:
            output_df.loc[date, symbol] = 1/4
        for symbol in top4_resistance:
            output_df.loc[date, symbol] = -1/4
    
    return output_df


def task1_Strategy5():
    train_data = pd.read_csv('train_data.csv')
    crossval_data = pd.read_csv('cross_val_data.csv')
    output_df = pd.DataFrame(0.0, index = range(3500, 4000), columns = range(20))

    data = pd.concat([train_data, crossval_data], ignore_index = True)
    for date in range(3500, 4000):
        k_metrics = []
        for symbol in range(20):
            symbol_data = data[data['Symbol'] == symbol].set_index('Date')['Close']
            #Last 14 closes
            closes_14 = symbol_data.loc[date - 14 : date - 1] if (date - 14 >= 0) else symbol_data.loc[: date - 1]
            if len(closes_14) < 14:
                k_metrics.append((symbol, 0))
                continue
            #Now we find the low and high stocks each for these 14 days
            low_14 = closes_14.min()
            high_14 = closes_14.max()
            close = symbol_data.get(date, np.isnan)
            if np.isnan(close) or np.isnan(low_14) or np.isnan(high_14) or low_14 == high_14: #If the lowest and highest values are same, then we will get a division by 0 error in later caluclations
                k_metrics.append((symbol, 0))
                continue
            k = 100 * (close - low_14) / (high_14 - low_14)
            k_metrics.append((symbol, k))
        #There might be some cases where the close for the date is the same as the lowest for the 14, so we will remove any metrics where k = 0
        k_metrics = [x for x in k_metrics if not np.isnan(x[1])] #Since the 2nd value of each tuple is the k metric
        #Sort k metrics in decreasing order
        k_metrics.sort(key = lambda x: -x[1])
        #We get the dates corresponding to the top and bottom 3 k-metrics
        top3 = [x[0] for x in k_metrics[:3]]
        bottom3 = [x[0] for x in k_metrics[-3:]]
        #Assign weights of -1/3 to top 3 stocks, and +1/3 to bottom 3 stocks, and 0 to the rest
        for symbol in top3:
            output_df.loc[date, symbol] = -1/3
        for symbol in bottom3:
            output_df.loc[date, symbol] = 1/3   
    return output_df


def task1():
    Strategy1 = task1_Strategy1()
    print(Strategy1.head(20))
    # Strategy2 = task1_Strategy2()
    # Strategy3 = task1_Strategy3()
    # Strategy4 = task1_Strategy4()
    # Strategy5 = task1_Strategy5()

    performanceStrategy1 = backtester_without_TC(Strategy1)
    print("Perfomance: ", performanceStrategy1)
    # performanceStrategy2 = backtester_without_TC(Strategy2)
    # performanceStrategy3 = backtester_without_TC(Strategy3)
    # performanceStrategy4 = backtester_without_TC(Strategy4)
    # performanceStrategy5 = backtester_without_TC(Strategy5)

    # output_df = pd.DataFrame({'Strategy1':performanceStrategy1, 'Strategy2': performanceStrategy2, 'Strategy3': performanceStrategy3, 'Strategy4': performanceStrategy4, 'Strategy5': performanceStrategy5})
    # output_df.to_csv('task1.csv')
    return



def task2():
    output_df_weights = pd.DataFrame()
    
    #Write your code here

    output_df_weights.to_csv('task2_weights.csv')
    results = backtester_without_TC(output_df_weights)
    df_performance = pd.DataFrame({'Net Returns': [results[0]], 'Sharpe Ratio': [results[1]]})
    df_performance.to_csv('task_2.csv')
    return



def calculate_turnover(weights_df):
    weights_diff_df = abs(weights_df-weights_df.shift(1))
    turnover_symbols = weights_diff_df.sum()
    turnover = turnover_symbols.sum()
    return turnover

def backtester_with_TC(weights_df):
    #Update path for data here
    data = pd.read_csv('file_path')

    weights_df = weights_df.fillna(0)

    turnover = calculate_turnover(weights_df)

    start_date = 3000
    end_date = 3499

    transaction_cost = (turnover * 0.01)

    df_returns = pd.DataFrame()

    for i in range(0,20):
        data_symbol = data[data['Symbol']==i]
        data_symbol = data_symbol['Close']
        data_symbol = data_symbol.reset_index(drop=True)   
        data_symbol = data_symbol/data_symbol.shift(1) - 1
        df_returns =  pd.concat([df_returns,data_symbol], axis=1, ignore_index=True)
    
    df_returns = df_returns.fillna(0)
    
    weights_df = weights_df.loc[start_date:end_date]    
    df_returns = df_returns.loc[start_date:end_date]

    df_returns = weights_df.mul(df_returns)

    initial_notional = 1
    notional = initial_notional

    returns = []

    for date in range(start_date,end_date+1):
        returns.append(df_returns.loc[date].values.sum())
        notional = notional * (1+returns[date-start_date])

    net_return = ((notional - transaction_cost - initial_notional)/initial_notional)*100
    sharpe_ratio = (pd.DataFrame(returns).mean().values[0] - (transaction_cost/(end_date-start_date+1)))/pd.DataFrame(returns).std().values[0]

    return [net_return, sharpe_ratio]



def task3():
    output_df_weights = pd.DataFrame()
    
    #Write your code here

    output_df_weights.to_csv('task3_weights.csv')
    results = backtester_with_TC(output_df_weights)
    df_performance = pd.DataFrame({'Net Returns': [results[0]], 'Sharpe Ratio': [results[1]]})
    df_performance.to_csv('task_3.csv')
    return



if __name__ == '__main__':
    task1()
    #task2()
    #task3()