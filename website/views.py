from datetime import datetime, timedelta
from django.shortcuts import render
from website.models import Assets,Strategies,EvaluationMetrics ,AssetsOHLCV
#from django_tut_2.settings import STRAT_EXAMPLE_IMAGE_PATH
from website.settings import STRAT_EXAMPLE_IMAGE_PATH
import os
import json

def read_text_file(filename):
    filepath = os.path.join('website/static/text_files/',filename)
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), filepath), 'r') as f:
        data = f.readlines()
    return data

def translate_strat_params(strategy_params):
    import json
    if isinstance(strategy_params,str):
        strategy_params = json.loads(strategy_params)

    #!!! DOESNT TAKE INTO ACCOUNT SIMPLE VS EXPONENTIAL MAs
    # exclude = ['use_slow_SMA','use_fast_SMA','use_SMA','use_bias_SMA']

    translation_dict = {'MA':'Simple Moving Average',
                        'ATR_num':'ATR Periods',
                        'ATR_mult':'ATR Multiple',
                        'R_mult_PT':'Profit Target Multiple',
                        'slow_MA':'Slow Simple Moving Average Moving Average',
                        'fast_MA':'Fast Simple Moving Average Moving Average',
                        'bias_MA':'Bias Simple Moving Average Moving Average',
                        'RSI_upper_bias':'RSI Upper Bias',
                        'RSI_lower_bias':'RSI Lower Bias',
                        'RSI_num':'RSI Periods',
                        'RVI_num':'RVI Periods',
                        'RVI_signal':'RVI Signal Periods',
                        'RVI_std_dev_num':'RVI Std Deviation Periods',
                        'RVI_std_dev_cutoff':'RVI Std Deviation Multiple',
                        'BB_window':'Bollinger Band Periods',
                        'BB_std_dev':'Standard Deviation',
                        'consec_candles':'Consecutive Candles',
                        'ATR_PP_chg_mult':'ATR Pivot Point Range'}

    replacement_dict = {}
    for k,v in strategy_params.items():
        try:
            replacement_dict[translation_dict[k]] = v
        except KeyError:
            pass

    return replacement_dict

##################################################################

def strategy_backtests(request, strat_name, asset_ticker):
    """
    http://127.0.0.1:8000/test/single-ma-strategy-version-1/eur_usd/
    http://127.0.0.1:8000/backtests/single-ma-strategy-version-1/eur_usd/
    Single MA Strategy Version 1
    NEED TO GET:
        - list of active strategies X
        - current strategy name, description X
        - list of active assets X
        - get current asset X
        - get timeframe X
        - get start and end dates X
        - query top 5 winning permuations for this month X
            - total realized R, strike rate, num trades X
        - query top 5 losing permutations for this month X
            - total realized R, strike rate, num trades X
        - query top 5 winning permuations for last month
            - total realized R, strike rate, num trades
        - query top 5 losing permutations for last month
            - total realized R, strike rate, num trades
    """

    def get_eval_results_for_month(start_date, end_date):

        # print('period start date',period_start_date)
        # print('period end date', period_end_date)

        eval_metrics_qs = EvaluationMetrics.objects.values(
            'strat_parameters_id',
            'total_realized_r',
            'strike_rate',
            'expectancy',
            'num_trades')

        eval_metrics_qs = eval_metrics_qs.filter(strat_id=current_strat_dict['strategy_id'],
                               asset_id=current_asset_dict['asset_id'],
                               period_end_date__date__lte=end_date,
                               period_start_date__date__gte=start_date)


        total_count = eval_metrics_qs.count()

        bottom_x_results_dict = eval_metrics_qs.order_by('total_realized_r')[:5]
        top_x_results_dict = eval_metrics_qs.order_by('-total_realized_r')[:5]

        return top_x_results_dict, bottom_x_results_dict,total_count

    ##########################################################

    # TODO consider making 2 queries for 1: just strategy names and 2: for the select strategy you want to avoid pulling extra data
    #                   (since your going to be putting example pics into the database)
    # TODO split assets in subgroups (i.e FOREX, Crypto, Equities), need to update models/migrations for "market" field

    strat_name = strat_name.replace('-','_').lower()
    asset_ticker = asset_ticker.upper()

    all_assets_list_dict = Assets.objects.all().values()
    current_asset_dict = [item for item in all_assets_list_dict if item['asset_ticker'] == asset_ticker][0]



    all_strats_list_dict = Strategies.objects.all().values()
    current_strat_dict = [item for item in all_strats_list_dict if item['strategy_reference'].lower() == strat_name][0]
    print(current_strat_dict)
    strats_sidebar_list = [item['strategy_reference'].replace('_', ' ').upper() for item in all_strats_list_dict]

    month_0_end = datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_0_start = (month_0_end - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_1_end = month_0_start
    month_1_start = (month_1_end - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    curr_month_top_results_dict, curr_month_bottom_results_dict,curr_month_perms_count = get_eval_results_for_month(start_date=month_0_start,
                                                                                             end_date=month_0_end)

    prev_month_top_results_dict, prev_month_bottom_results_dict,prev_month_perms_count = get_eval_results_for_month(start_date=month_1_start,
                                                                                                                    end_date=month_1_end)

    strat_image_path = STRAT_EXAMPLE_IMAGE_PATH+current_strat_dict['strategy_reference'].lower()+'.png'

    ###########################################################################################################

    context = {'strats_sidebar_list':strats_sidebar_list,
               'sidebar_default_asset':all_assets_list_dict[0]['asset_ticker'],
               'all_assets_list_dict':all_assets_list_dict,
               'current_strat_dict':current_strat_dict,
               'current_asset_dict':current_asset_dict,
               'timeframe':'1 Hour Candles',
               'strat_image_path':strat_image_path,
               'current_month_data':{'start_date':str(month_0_start).split(' ')[0],
                                     'end_date':str(month_0_end).split(' ')[0],
                                     'top_winning_strats':curr_month_top_results_dict,
                                     'top_losing_strats':curr_month_bottom_results_dict,
                                     'count':curr_month_perms_count},

               'previous_month_data':{'start_date':str(month_1_start).split(' ')[0],
                                     'end_date':str(month_1_end).split(' ')[0],
                                     'top_winning_strats':prev_month_top_results_dict,
                                     'top_losing_strats':prev_month_bottom_results_dict,
                                      'count':prev_month_perms_count},
               }

    return render(request,'backtests.html',context)

def strategy_backtest_default(request):
    """
    meant to appear as the "backtests" link on the navbar and get the first strategy and asset items as a default
    to take you to the backtests results page
    """
    all_strats_qs = Strategies.objects.filter(pk=1).values('strategy_reference')
    default_strat = [item['strategy_reference'] for item in all_strats_qs][0]

    all_assets_qs = Assets.objects.filter(pk=1).values('asset_ticker')
    default_asset = [item['asset_ticker'] for item in all_assets_qs][0]

    return strategy_backtests(request,default_strat,default_asset)

def individual_results(request,strat_name,asset_ticker,perm_id):
    """
    need
    strategies
        - list of strategies for sidebar
        - strat name
        - strat description
    evaluation_metrics
        - * NEED TO FIGURE OUT HOW TO DISPLAY CHARTS
        - eval metrics for strat id, asset id, timeframe, strat params/strat id
        - list of all trades for that strategy

    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    from website.indicators import technical_indicators as ti

    def create_trades_chart(candle_data_df, trade_data, strat_params):  # ):#,start_date,end_date):

        def create_entry_exit_dots(fig, trade_info):

            x = trade_info['entry_datetime']
            y = trade_info['entry_price']
            fig.add_trace(
                go.Scatter(x=[x], y=[y], marker=dict(color='green', size=9),
                           name=str(trade_info['trade_num']) + ' ' + trade_info['bias'] + ' entry'),
                row=1, col=1)

            x = trade_info['exit_datetime']
            y = trade_info['exit_price']
            fig.add_trace(
                go.Scatter(x=[x], y=[y], marker=dict(color='red', size=9),
                           name=str(trade_info['trade_num']) + ' exit'), row=1, col=1)

            return fig

        def chart_indicators(candle_df, strat_params, strat_reference):
            # TODO need to figure out a way to chart relevant stuff on the chart depending on what is asked
            #       could just make pre fabbed chart functions for each strat put it in a dict and set keys=strat ids
            #       each function returns a fig object that has all relevant data charted to it



            def graph_overlay(fig, data_df, y_name, name, color):
                fig.add_trace(
                    go.Scatter(x=data_df['Date_Time'],
                               y=data_df[y_name],
                               mode='lines',
                               name=name,
                               line=dict(color=color)),
                    row=1, col=1)

                return fig

            ###############################################################

            def basic_X_MA_crossover_V1_INDICATORS(data_df, params):
                # labels SMA/EMA
                if params['use_SMA'] == 1:
                    data_df = ti.simple_moving_average(df=data_df, MA_num=params['MA'])
                    MA_header = str(params['MA']) + 'SMA'
                else:
                    data_df = ti.exponential_moving_average(df=data_df, MA_num=params['MA'])
                    MA_header = str(params['MA']) + 'EMA'

                # get list of all indicators and drop rows of those columns with NA vals
                orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

                data_df = data_df.dropna(subset=indicator_col_names)
                data_df = data_df.reset_index(drop=True)

                ##################################################################################

                fig = make_subplots(rows=1, cols=1)

                # graph candlestick
                fig.add_trace(
                    go.Candlestick(x=candle_data_df['Date_Time'], open=candle_data_df['Open'],
                                   high=candle_data_df['High'],
                                   low=candle_data_df['Low'], close=candle_data_df['Close'], name='Candlestick Data'),
                    row=1, col=1)

                # graph MA
                fig = graph_overlay(fig=fig,
                                    data_df=data_df,
                                    y_name=MA_header,
                                    name=MA_header,
                                    color='purple')

                return fig

            def basic_X_MA_crossover_V2_INDICATORS(data_df, strat_params):
                # SMA/EMA
                if params['use_SMA'] == 1:
                    data_df = ti.simple_moving_average(df=data_df, MA_num=params['MA'])
                    MA_header = str(params['MA']) + 'SMA'
                else:
                    data_df = ti.exponential_moving_average(df=data_df, MA_num=params['MA'])
                    MA_header = str(params['MA']) + 'EMA'

                # get list of all indicators and drop rows of those columns with NA vals
                orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

                data_df = data_df.dropna(subset=indicator_col_names)
                data_df = data_df.reset_index(drop=True)

                ##################################################################################

                fig = make_subplots(rows=1, cols=1)

                # graph candlestick
                fig.add_trace(
                    go.Candlestick(x=candle_data_df['Date_Time'], open=candle_data_df['Open'],
                                   high=candle_data_df['High'],
                                   low=candle_data_df['Low'], close=candle_data_df['Close'], name='Candlestick Data'),
                    row=1, col=1)

                # graph MA
                fig = graph_overlay(fig=fig,
                                    data_df=data_df,
                                    y_name=MA_header,
                                    name=MA_header,
                                    color='purple')

                return fig

            def basic_XY_MA_crossover_V1_INDICATORS(data_df, strat_params):
                # SMA/EMA
                if params['use_slow_SMA'] == 1:
                    data_df = ti.simple_moving_average(df=data_df, MA_num=params['slow_MA'])
                    slow_MA_header = str(params['slow_MA']) + 'SMA'
                else:
                    data_df = ti.exponential_moving_average(df=data_df, MA_num=params['slow_MA'])
                    slow_MA_header = str(params['slow_MA']) + 'EMA'

                if params['use_fast_SMA'] == 1:
                    data_df = ti.simple_moving_average(df=data_df, MA_num=params['fast_MA'])
                    fast_MA_header = str(params['fast_MA']) + 'SMA'
                else:
                    data_df = ti.exponential_moving_average(df=data_df, MA_num=params['fast_MA'])
                    fast_MA_header = str(params['fast_MA']) + 'EMA'

                # get list of all indicators and drop rows of those columns with NA vals
                orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

                data_df = data_df.dropna(subset=indicator_col_names)
                data_df = data_df.reset_index(drop=True)

                ##################################################################################

                fig = make_subplots(rows=1, cols=1)

                # graph candlestick
                fig.add_trace(
                    go.Candlestick(x=candle_data_df['Date_Time'], open=candle_data_df['Open'],
                                   high=candle_data_df['High'],
                                   low=candle_data_df['Low'], close=candle_data_df['Close'], name='Candlestick Data'),
                    row=1, col=1)

                # graph MAs
                fig = graph_overlay(fig=fig,
                                    data_df=data_df,
                                    y_name=fast_MA_header,
                                    name=fast_MA_header,
                                    color='purple')

                fig = graph_overlay(fig=fig,
                                    data_df=data_df,
                                    y_name=slow_MA_header,
                                    name=slow_MA_header,
                                    color='orange')

                return fig

            def XYZ_MA_crossover_V1_INDICATORS(data_df, strat_params):
                # slow MA
                if params['use_slow_SMA'] == 1:
                    data_df = ti.simple_moving_average(df=data_df, MA_num=params['slow_MA'])
                    slow_MA_header = str(params['slow_MA']) + 'SMA'
                else:
                    data_df = ti.exponential_moving_average(df=data_df, MA_num=params['slow_MA'])
                    slow_MA_header = str(params['slow_MA']) + 'EMA'

                # fast MA
                if params['use_fast_SMA'] == 1:
                    data_df = ti.simple_moving_average(df=data_df, MA_num=params['fast_MA'])
                    fast_MA_header = str(params['fast_MA']) + 'SMA'
                else:
                    data_df = ti.exponential_moving_average(df=data_df, MA_num=params['fast_MA'])
                    fast_MA_header = str(params['fast_MA']) + 'EMA'

                # bias MA
                if params['use_bias_SMA'] == 1:  # use SMA
                    data_df = ti.simple_moving_average(df=data_df, MA_num=params['bias_MA'])
                    bias_MA_header = str(params['bias_MA']) + 'SMA'
                else:  # use EMA
                    data_df = ti.exponential_moving_average(df=data_df, MA_num=params['bias_MA'])
                    bias_MA_header = str(params['bias_MA']) + 'EMA'

                # get list of all indicators and drop rows of those columns with NA vals
                orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

                data_df = data_df.dropna(subset=indicator_col_names)
                data_df = data_df.reset_index(drop=True)

                ##################################################################################

                fig = make_subplots(rows=1, cols=1)

                # graph candlestick
                fig.add_trace(
                    go.Candlestick(x=candle_data_df['Date_Time'], open=candle_data_df['Open'],
                                   high=candle_data_df['High'],
                                   low=candle_data_df['Low'], close=candle_data_df['Close'], name='Candlestick Data'),
                    row=1, col=1)

                # graph MAs
                fig = graph_overlay(fig=fig,
                                    data_df=data_df,
                                    y_name=fast_MA_header,
                                    name=fast_MA_header,
                                    color='purple')

                fig = graph_overlay(fig=fig,
                                    data_df=data_df,
                                    y_name=slow_MA_header,
                                    name=slow_MA_header,
                                    color='orange')

                fig = graph_overlay(fig=fig,
                                    data_df=data_df,
                                    y_name=bias_MA_header,
                                    name=bias_MA_header,
                                    color='red')

                return fig

            # TODO need to do some custom stuff
            def RVI_crossover_V1_INDICATORS(data_df, strat_params):
                # RVI
                data_df = ti.RVI(df=data_df, periods=params['RVI_num'],
                                         signal_MA=params['RVI_signal'])
                RVI_header = str(params['RVI_num']) + 'RVI'
                RVI_signal_header = str(params['RVI_num']) + 'RVI_signal'

                # get list of all indicators and drop rows of those columns with NA vals
                orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

                data_df = data_df.dropna(subset=indicator_col_names)
                data_df = data_df.reset_index(drop=True)

                ##################################################################################

                fig = make_subplots(rows=2, cols=1)

                # graph candlestick
                fig.add_trace(
                    go.Candlestick(x=candle_data_df['Date_Time'], open=candle_data_df['Open'],
                                   high=candle_data_df['High'],
                                   low=candle_data_df['Low'], close=candle_data_df['Close'], name='Candlestick Data'),
                    row=1, col=1)

                # graph MAs
                fig.add_trace(
                    go.Scatter(x=data_df['Date_Time'],
                               y=data_df[fast_MA_header],
                               mode='lines',
                               name=MA_header,
                               line=dict(color='orange')),
                    row=1, col=1)

                fig.add_trace(
                    go.Scatter(x=data_df['Date_Time'],
                               y=data_df[slow_MA_header],
                               mode='lines',
                               name=MA_header,
                               line=dict(color='blue')),
                    row=1, col=1)

                fig.add_trace(
                    go.Scatter(x=data_df['Date_Time'],
                               y=data_df[bias_MA_header],
                               mode='lines',
                               name=MA_header,
                               line=dict(color='green')),
                    row=1, col=1)

                return fig

            # TODO need to do some custom stuff
            def RVI_crossover_V2_INDICATORS(data_df, strat_params):
                # RVI
                data_df = ti.RVI(df=data_df, periods=params['RVI_num'],
                                         signal_MA=params['RVI_signal'])
                df_col_headers.RVI_header = str(params['RVI_num']) + 'RVI'
                df_col_headers.RVI_signal_header = str(params['RVI_num']) + 'RVI_signal'

                # ATR
                data_df = ti.average_true_range(df=data_df, window=params['ATR_num'])
                df_col_headers.ATR_header = str(params['ATR_num']) + 'ATR'

                # get list of all indicators and drop rows of those columns with NA vals
                orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

                data_df = data_df.dropna(subset=indicator_col_names)
                data_df = data_df.reset_index(drop=True)
                return data_df

            def Basic_Bollinger_Bands_V1_INDICATORS(data_df, strat_params):
                # BB
                df = ti.bollinger_bands(df=df, window=params['BB_window'],
                                                std_dev=params['BB_std_dev'])

                # get list of all indicators and drop rows of those columns with NA vals
                orig_col_names = ['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                indicator_col_names = [col for col in df.columns.values.tolist() if col not in orig_col_names]

                df = df.dropna(subset=indicator_col_names)
                df = df.reset_index(drop=True)

                ##################################################################################

                fig = make_subplots(rows=1, cols=1)

                # graph candlestick
                fig.add_trace(
                    go.Candlestick(x=candle_data_df['Date_Time'], open=candle_data_df['Open'],
                                   high=candle_data_df['High'],
                                   low=candle_data_df['Low'], close=candle_data_df['Close'], name='Candlestick Data'),
                    row=1, col=1)

                # graph BB
                fig = graph_overlay(fig=fig,
                                    data_df=data_df,
                                    y_name='BB mean',
                                    name='BB mean',
                                    color='orange')

                fig = graph_overlay(fig=fig,
                                    data_df=data_df,
                                    y_name='BB upper',
                                    name='BB upper',
                                    color='purple')

                fig = graph_overlay(fig=fig,
                                    data_df=data_df,
                                    y_name='BB lower',
                                    name='BB lower',
                                    color='purple')

                return fig

            def basic_heiken_ashi_V1_INDICATORS(data_df, strat_params):
                # candle_num
                data_df = data_df.reset_index()

                # heiken ashi
                data_df = ti.heiken_ashi(df=data_df)
                data_df['prev_HA_red_green'] = data_df['HA_red_green'].shift(1)
                data_df['is_pivot_point'] = np.where(data_df['prev_HA_red_green'] != data_df['HA_red_green'], 1, 0)

                data_df["consec_red_candles"] = data_df.groupby(
                    (data_df["prev_HA_red_green"] == 'green').cumsum()).cumcount()
                data_df["consec_green_candles"] = data_df.groupby(
                    (data_df["prev_HA_red_green"] == 'red').cumsum()).cumcount()

                # get list of all indicators and drop rows of those columns with NA vals
                orig_col_names = ['Date_Time', 'HA Open', 'HA High', 'HA Low', 'HA Close', 'Volume']
                indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

                data_df = data_df.dropna(subset=indicator_col_names)
                data_df = data_df.reset_index(drop=True)

                ##############################################################################

                fig = make_subplots(rows=1, cols=1)

                # graph candlestick
                fig.add_trace(
                    go.Candlestick(x=candle_data_df['Date_Time'], open=candle_data_df['HA Open'],
                                   high=candle_data_df['HA High'],
                                   low=candle_data_df['HA Low'], close=candle_data_df['HA Close'],
                                   name='Candlestick Data'),
                    row=1, col=1)

                return fig

            def basic_heiken_ashi_V2_INDICATORS(data_df, strat_params):
                # candle_num
                data_df = data_df.reset_index()

                # heiken ashi
                data_df = ti.heiken_ashi(df=data_df)

                data_df['prev_HA_red_green'] = data_df['HA_red_green'].shift(1)
                data_df['is_pivot_point'] = np.where(data_df['prev_HA_red_green'] != data_df['HA_red_green'], 1, 0)

                data_df['pivot_price'] = np.where(data_df['HA_red_green'] == 'green', data_df['HA Open'],
                                                  data_df['HA Close'])

                data_df["consec_red_candles"] = data_df.groupby(
                    (data_df["prev_HA_red_green"] == 'green').cumsum()).cumcount()
                data_df["consec_green_candles"] = data_df.groupby(
                    (data_df["prev_HA_red_green"] == 'red').cumsum()).cumcount()

                # get list of all indicators and drop rows of those columns with NA vals
                orig_col_names = ['Date_Time', 'HA Open', 'HA High', 'HA Low', 'HA Close', 'Volume']
                indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

                data_df = data_df.dropna(subset=indicator_col_names)
                data_df = data_df.reset_index(drop=True)

                ##############################################################################

                fig = make_subplots(rows=1, cols=1)

                # graph candlestick
                fig.add_trace(
                    go.Candlestick(x=candle_data_df['Date_Time'], open=candle_data_df['HA Open'],
                                   high=candle_data_df['HA High'],
                                   low=candle_data_df['HA Low'], close=candle_data_df['HA Close'],
                                   name='Candlestick Data'),
                    row=1, col=1)

                return fig

            def basic_heiken_ashi_V3_INDICATORS(data_df, strat_params):
                # candle_num
                data_df = data_df.reset_index()

                # heiken ashi
                data_df = ti.heiken_ashi(df=data_df)

                data_df['prev_HA_red_green'] = data_df['HA_red_green'].shift(1)
                data_df['is_pivot_point'] = np.where(data_df['prev_HA_red_green'] != data_df['HA_red_green'], 1, 0)

                data_df['pivot_price'] = np.where(data_df['HA_red_green'] == 'green', data_df['HA Open'],
                                                  data_df['HA Close'])

                data_df["consec_red_candles"] = data_df.groupby(
                    (data_df["prev_HA_red_green"] == 'green').cumsum()).cumcount()
                data_df["consec_green_candles"] = data_df.groupby(
                    (data_df["prev_HA_red_green"] == 'red').cumsum()).cumcount()

                # get list of all indicators and drop rows of those columns with NA vals
                orig_col_names = ['Date_Time', 'HA Open', 'HA High', 'HA Low', 'HA Close', 'Volume']
                indicator_col_names = [col for col in data_df.columns.values.tolist() if col not in orig_col_names]

                data_df = data_df.dropna(subset=indicator_col_names)
                data_df = data_df.reset_index(drop=True)

                ##############################################################################

                fig = make_subplots(rows=1, cols=1)

                # graph candlestick
                fig.add_trace(
                    go.Candlestick(x=candle_data_df['Date_Time'], open=candle_data_df['HA Open'],
                                   high=candle_data_df['HA High'],
                                   low=candle_data_df['HA Low'], close=candle_data_df['HA Close'],
                                   name='Candlestick Data'),
                    row=1, col=1)

                return fig

            ######################################################################################################

            offset_candles = max([item for item in strat_params.values()])

            # df = helper_functions.slice_backtest_ohlcv_df(df=df,
            #                                               offset_candles=offset_candles,
            #                                               start_date=start_date,
            #                                               end_date=end_date)

            strats_dict = {'basic_X_MA_crossover_V1': basic_X_MA_crossover_V1_INDICATORS,
                           'basic_X_MA_crossover_V2': basic_X_MA_crossover_V2_INDICATORS,
                           'basic_XY_MA_crossover_V1': basic_XY_MA_crossover_V1_INDICATORS,
                           'XYZ_MA_crossover_V1': XYZ_MA_crossover_V1_INDICATORS,
                           'RVI_crossover_V1': RVI_crossover_V1_INDICATORS,
                           'RVI_crossover_V2': RVI_crossover_V2_INDICATORS,
                           'Basic_Bollinger_Bands_V1': Basic_Bollinger_Bands_V1_INDICATORS,
                           'Intermediate_Bollinger_Bands_V1': basic_heiken_ashi_V1_INDICATORS,
                           'basic_heiken_ashi_V1': basic_heiken_ashi_V1_INDICATORS,
                           'basic_heiken_ashi_V2': basic_heiken_ashi_V2_INDICATORS,
                           'basic_heiken_ashi_V3': basic_heiken_ashi_V3_INDICATORS}

            fig = strats_dict[strat_reference](candle_df, strat_params)

            return fig

        ###############################################################################

        fig = chart_indicators(candle_df=candle_data_df,
                               strat_params=strat_params,
                               strat_reference='basic_X_MA_crossover_V1')

        for trade in trade_data:
            fig = create_entry_exit_dots(fig=fig, trade_info=trade)

        fig.update_layout(title='Candlestick With Trades Display',
                          yaxis_title='Price',
                          xaxis_title='Date',
                          xaxis_rangeslider_visible=False)

        return fig.to_html()

    def create_equity_curve_chart(candle_data_df, trade_data):
        trade_data_df = pd.DataFrame(trade_data)

        # trade_data_df['dollar_return'] = trade_data_df['exit_price'] - trade_data_df['entry_price']
        #
        # trade_data_df['R_realized'] = np.where(trade_data_df['bias'] == 'buy',
        #                                        (trade_data_df['dollar_return'])/(trade_data_df['entry_price'] - trade_data_df['risk_price']),
        #                                        (trade_data_df['dollar_return'])/(trade_data_df['risk_price'] - trade_data_df['entry_price']))

        # trade_data_df['R_realized_cum'] = trade_data_df['R_realized'].cumsum()

        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Candlestick(x=candle_data_df['Date_Time'], open=candle_data_df['Open'], high=candle_data_df['High'],
                           low=candle_data_df['Low'], close=candle_data_df['Close'],
                           name='Candlestick Data'), row=1, col=1, secondary_y=False)

        fig.add_trace(
            go.Scatter(x=trade_data_df['exit_datetime'],
                       y=trade_data_df['running_total_R'],
                       marker=dict(color='orange', size=9),
                       name='R Return'),
            row=1, col=1, secondary_y=True)

        fig.update_layout(title='Candlestick With Equity Curve',
                          xaxis_title='Date',
                          xaxis_rangeslider_visible=False)

        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="R Return", secondary_y=True)

        return fig.to_html()

    def create_MFE_MAE_chart(perm_dict):

        def bin_df(base_df, bins=32):

            def get_x_axis(df, title):
                df = df[[title]]

                df[title + '_bins'] = pd.cut(df[title], bins)
                df = df.groupby(by=[title + '_bins']).count()
                df = df.reset_index()

                df[title + '_bins'] = df[title + '_bins'].astype('string')
                df[title + '_bins'] = df[title + '_bins'].str.split(',').str[0]
                df[title + '_bins'] = df[title + '_bins'].str.replace('(', '').str.strip()
                df[title + '_bins'] = df[title + '_bins'].astype('float')

                if title == 'MAE':
                    df = df[::-1]
                    df = df.reset_index(drop=True)
                    df = df.reset_index()
                    df['index'] += 1
                    df['index'] *= -1
                else:
                    df = df.reset_index()
                    df['index'] += 1

                return df

            #######################################################

            base_df = base_df[['MAE', 'MFE']]
            base_df['MAE'] = base_df['MAE'] * -1

            MAE_df = get_x_axis(df=base_df, title='MAE')
            MFE_df = get_x_axis(df=base_df, title='MFE')

            return MAE_df, MFE_df

        def chart_MAE(MAE_df, chart_type):

            fig = make_subplots(rows=1, cols=1)

            max_y = MAE_df['MAE'].max() + 1

            # MAE
            fig.add_trace(go.Bar(x=MAE_df['MAE_bins'], y=MAE_df['MAE'], name='MAE Distribution',
                                 marker_color='orange'),
                          row=1, col=1)

            # -1R cutoff
            fig.add_trace(go.Scatter(x=[-1, -1], y=[-1, max_y], mode='lines', line=dict(color='red', width=8),
                                     name='1R Risk Price'), row=1, col=1)

            fig.add_annotation(x=-1, y=max_y,
                               xref="x",
                               text="-1R Cutoff",
                               textangle=45,
                               showarrow=True,
                               arrowhead=1,
                               #font=dict(size=24)
                               )

            ##################################################################

            # Entry Price
            fig.add_trace(go.Scatter(x=[0, 0], y=[-1, max_y], mode='lines', line=dict(color='black', width=8),
                                     name='Entry Price'), row=1, col=1)

            fig.add_annotation(x=0, y=max_y,
                               xref="x",
                               text="Entry Price",
                               textangle=45,
                               showarrow=True,
                               arrowhead=1,
                               #font=dict(size=24)
                               )

            ##################################################################

            if chart_type == 'winner':
                # avg winner MAE
                fig.add_trace(go.Scatter(x=[perm_dict['winners_avg_mae'], perm_dict['winners_avg_mae']],
                                         y=[-1, max_y], mode='lines', line=dict(color='maroon', width=8),
                                         name='Avg Winner MAE'), row=1, col=1)

                fig.add_annotation(x=perm_dict['winners_avg_mae'], y=max_y,
                                   xref="x",
                                   text="Avg Winner MAE",
                                   textangle=45,
                                   showarrow=True,
                                   arrowhead=1,
                                   #font=dict(size=24)
                                   )

                fig.update_layout(title="Winners MAE",
                                  xaxis_title="R Levels")

            if chart_type == 'loser':
                # avg loser MAE
                fig.add_trace(go.Scatter(x=[perm_dict['losers_avg_mae'], perm_dict['losers_avg_mae']],
                                         y=[-1, max_y], mode='lines', line=dict(color='maroon', width=8),
                                         name='Avg Loser MAE'), row=1, col=1)

                fig.add_annotation(x=perm_dict['losers_avg_mae'], y=max_y,
                                   xref="x",
                                   text="Avg Loser MAE",
                                   textangle=45,
                                   showarrow=True,
                                   arrowhead=1,
                                   #font=dict(size=24)
                                   )

                # Avg R loser
                fig.add_trace(go.Scatter(x=[perm_dict['avg_loser'], perm_dict['avg_loser']],
                                         y=[-1, max_y], mode='lines', line=dict(color='forestgreen', width=8),
                                         name='Avg Loser'), row=1, col=1)

                fig.add_annotation(x=perm_dict['avg_loser'], y=max_y,
                                   xref="x",
                                   text="Avg Loser",
                                   textangle=45,
                                   showarrow=True,
                                   arrowhead=1,
                                   #font=dict(size=24)
                                   )

                fig.update_layout(title="Losers MAE",
                                  xaxis_title="R Levels")

            # fig.layout.yaxis2.update(showticklabels=False)
            #fig.update_layout(font=dict(size=32))

            return fig.to_html()

        def chart_MFE(MFE_df, chart_type):
            fig = make_subplots(rows=1, cols=1)

            max_y = MFE_df['MFE'].max() + 1

            # MFE
            fig.add_trace(go.Bar(x=MFE_df['MFE_bins'], y=MFE_df['MFE'],
                                 marker_color='steelblue', name='MFE Distribution'),row=1, col=1)

            # Entry Price
            fig.add_trace(go.Scatter(x=[0, 0], y=[-1, max_y], mode='lines', line=dict(color='black', width=8),
                                     name='Entry Price'), row=1, col=1)

            fig.add_annotation(x=0, y=max_y,
                               xref="x",
                               text="Entry Price",
                               textangle=45,
                               showarrow=True,
                               arrowhead=1,
                               #font=dict(size=24)
                               )

            #######################################################################

            if chart_type == 'winner':
                # avg winner MFE
                fig.add_trace(go.Scatter(x=[perm_dict['winners_avg_mfe'], perm_dict['winners_avg_mfe']],
                                         y=[-1, max_y], mode='lines+text', line=dict(color='deeppink', width=8),
                                         name='Avg Winner MFE'), row=1, col=1)

                fig.add_annotation(x=perm_dict['winners_avg_mfe'], y=max_y,
                                   xref="x",
                                   text="Avg Winner MFE",
                                   textangle=45,
                                   showarrow=True,
                                   arrowhead=1,
                                   #font=dict(size=24)
                                   )

                #######################################################################

                # Avg R Winner
                fig.add_trace(go.Scatter(x=[perm_dict['avg_winner'], perm_dict['avg_winner']],
                                         y=[-1, max_y], mode='lines', line=dict(color='forestgreen', width=8),
                                         name='Avg Winner'), row=1, col=1)

                fig.add_annotation(x=perm_dict['avg_winner'], y=max_y,
                                   xref="x",
                                   text="Avg Winner",
                                   textangle=45,
                                   showarrow=True,
                                   arrowhead=1,
                                   #font=dict(size=24)
                                   )

                fig.update_layout(title="Winners MFE",
                                  xaxis_title="R Levels")

            if chart_type == 'loser':
                # avg loser MFE
                fig.add_trace(go.Scatter(x=[perm_dict['losers_avg_mfe'], perm_dict['losers_avg_mfe']],
                                         y=[-1, max_y], mode='lines', line=dict(color='seagreen', width=8),
                                         name='Avg Loser MFE'), row=1, col=1)

                fig.add_annotation(x=perm_dict['losers_avg_mfe'], y=max_y,
                                   xref="x",
                                   text="Avg Loser MFE",
                                   textangle=45,
                                   showarrow=True,
                                   arrowhead=1,
                                   #font=dict(size=24)
                                   )

                fig.update_layout(title="Losers MFE",
                                  xaxis_title="R Levels")

            return fig.to_html()

        #######################################################

        if not isinstance(perm_dict['trades_list'], list):
            perm_dict['trades_list'] = json.loads(perm_dict['trades_list'])

        trades_data_df = pd.DataFrame(perm_dict['trades_list'])

        winning_trades_df = trades_data_df.loc[trades_data_df['R_realized'] >= 0]
        losing_trades_df = trades_data_df.loc[trades_data_df['R_realized'] <= 0]

        winning_trades_MAE_df, winning_trades_MFE_df = bin_df(base_df=winning_trades_df)
        losing_trades_MAE_df, losing_trades_MFE_df = bin_df(base_df=losing_trades_df)

        winning_MFE_fig = chart_MFE(MFE_df=winning_trades_MFE_df, chart_type='winner')
        winning_MAE_fig = chart_MAE(MAE_df=winning_trades_MAE_df, chart_type='winner')
        losing_MFE_fig = chart_MFE(MFE_df=losing_trades_MFE_df, chart_type='loser')
        losing_MAE_fig = chart_MAE(MAE_df=losing_trades_MAE_df, chart_type='loser')

        ret_dict = {'winners_MFE_chart':winning_MFE_fig,
                    'winners_MAE_chart':winning_MAE_fig,
                    'losers_MFE_chart': losing_MFE_fig,
                    'losers_MAE_chart':losing_MAE_fig}

        return ret_dict

    def convert_json(x):
        if isinstance(x, str):
            return json.loads(x)
        return x

    def translate_strat_params(strategy_params):
        strategy_params = convert_json(x=strategy_params)

        # !!! DOESNT TAKE INTO ACCOUNT SIMPLE VS EXPONENTIAL MAs
        # exclude = ['use_slow_SMA','use_fast_SMA','use_SMA','use_bias_SMA']

        translation_dict = {'MA': 'Moving Average',
                            'ATR_num': 'ATR Periods',
                            'ATR_mult': 'ATR Multiple',
                            'R_mult_PT': 'Profit Target Multiple',
                            'slow_MA': 'Slow Moving Average',
                            'fast_MA': 'Fast Moving Average',
                            'bias_MA': 'Bias Moving Average',
                            'RSI_upper_bias': 'RSI Upper Bias',
                            'RSI_lower_bias': 'RSI Lower Bias',
                            'RSI_num': 'RSI Periods',
                            'RVI_num': 'RVI Periods',
                            'RVI_signal': 'RVI Signal Periods',
                            'RVI_std_dev_num': 'RVI Std Deviation Periods',
                            'RVI_std_dev_cutoff': 'RVI Std Deviation Multiple',
                            'BB_window': 'Bollinger Band Periods',
                            'BB_std_dev': 'Standard Deviation',
                            'consec_candles': 'Consecutive Candles',
                            'ATR_PP_chg_mult': 'ATR Pivot Point Range'}

        replacement_dict = {}
        for k, v in strategy_params.items():
            try:
                replacement_dict[translation_dict[k]] = v
            except KeyError:
                pass

        return replacement_dict

    def translate_eval_metrics(metrics):
        ret_val = {}
        for k,v in metrics.items():
            k = k.replace('_',' ')
            k = k.split(' ')
            k = ' '.join([p.capitalize() for p in k])
            ret_val[k] = v
        return ret_val

    def clean_DB_to_DF(df):
        keep_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        rename_col = {}
        for col in keep_cols:
            if col == 'datetime':
                rename_col[col] = 'Date_Time'
            else:
                rename_col[col] = col.capitalize()

        df = df[keep_cols]
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns=rename_col)
        return df

    def filter_evaluation_metrics(all_metrics):
        exclude = ['strat_parameters_id', 'strat_id', 'asset_id', 'period_end_date', 'period_start_date', 'timeframe',
                   'strategy_params', 'evaluation_metricscol', 'trades_list']
        return {k:v for (k,v) in all_metrics.items() if k not in exclude}

    def filter_candle_df_by_dates(df,start_date,end_date):

        for d in [start_date,end_date]:
            if not isinstance(d,datetime):
                start_date = datetime.strptime(d,'%Y-%m-%d %H:%M:%S')

        df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        df = df.loc[(df['Date_Time'] >= start_date) & (df['Date_Time'] <= end_date)]
        return df

    ##########################################################################################################
    print(strat_name)
    print(asset_ticker)
    print(perm_id)
    print('---------------------------------------------------------------')
    print('---------------------------------------------------------------')
    print('---------------------------------------------------------------')

    all_strats_qs = Strategies.objects.all().values('strategy_reference')
    all_strats = [item['strategy_reference'].replace('_', ' ').upper() for item in all_strats_qs]

    default_asset_qs = Assets.objects.filter(pk=1).values('asset_ticker')

    # data for x permutation
    perm_details_dict = EvaluationMetrics.objects.filter(pk=perm_id).values()[0]

    # get candle data and put into dataframe
    asset_ohlcv_data_qs = AssetsOHLCV.objects.filter(asset_ticker_identification=perm_details_dict['asset_id'],
                                               datetime__range=(perm_details_dict['period_start_date'],
                                                                perm_details_dict['period_end_date']))

    asset_ohlcv_data_qs = AssetsOHLCV.objects.all()
    candle_data_df = pd.DataFrame(asset_ohlcv_data_qs.values('datetime', 'open', 'high', 'low', 'close', 'volume'))
    candle_data_df = clean_DB_to_DF(df=candle_data_df)
    candle_data_df = filter_candle_df_by_dates(df=candle_data_df,
                                               start_date=perm_details_dict['period_start_date'],
                                               end_date=perm_details_dict['period_end_date'])

    current_strat_dict = Strategies.objects.filter(strategy_id=perm_details_dict['strat_id']).values()[0]
    print(current_strat_dict)
    # get info the strategy responsible for that permutation
    # current_strat_dict = list(filter(lambda strat: strat['strategy_id'] == perm_details_dict['strat_id'], strategy_dict))[0]

    # set up the annotated candlestick chart
    candle_chart = create_trades_chart(candle_data_df=candle_data_df,
                                   trade_data=convert_json(x=perm_details_dict['trades_list']),
                                   strat_params=convert_json(x=perm_details_dict['strategy_params']))

    # set up the equity curve chart
    equity_curve_chart = create_equity_curve_chart(candle_data_df=candle_data_df,
                                                  trade_data=convert_json(x=perm_details_dict['trades_list']))

    MAE_MFE_charts = create_MFE_MAE_chart(perm_dict=perm_details_dict)

    # convert strat params into a more readable format
    strategy_params = translate_strat_params(strategy_params=perm_details_dict['strategy_params'])

    # filters out metrics that arent relevant to strategy evaluation
    evaluation_metrics = filter_evaluation_metrics(all_metrics=perm_details_dict)
    evaluation_metrics = translate_eval_metrics(metrics=evaluation_metrics)
    # print(evaluation_metrics)
    ####################################################################################################################

    context = {'strats_sidebar_list':all_strats,
               'sidebar_default_asset': default_asset_qs[0]['asset_ticker'],
               'start_date':str(perm_details_dict['period_start_date']).split(' ')[0],
               'end_date':str(perm_details_dict['period_end_date']).split(' ')[0],
               'current_strat_dict':current_strat_dict,
               'strategy_params':strategy_params,
               'trades_list_titles':['Trade Num','Bias','Entry Price','Risk Price','Exit Price',
                                     'Entry Date','Exit Date','R Profit','Cumulative R'],
               'trades_list':perm_details_dict['trades_list'],
               'evaluation_metrics':evaluation_metrics,
               'candle_chart':candle_chart,
               'equity_curve_chart':equity_curve_chart,
               'excursion_charts':MAE_MFE_charts}

    # #print('all_strats_dict',all_strategies_qs)
    # print('------------------------------------------------')
    # print('current strat dict', current_strat_dict)
    # print('------------------------------------------------')
    # print('perm details', perm_details_dict)
    # print('------------------------------------------------')

    return render(request, 'individual_results.html',context)


def interpreting_results(request):
    all_strats_qs = Strategies.objects.all().values('strategy_reference')
    all_strats = [item['strategy_reference'].replace('_', ' ').upper() for item in all_strats_qs]

    default_asset_qs = Assets.objects.filter(pk=1).values('asset_ticker')
    context = {'strats_sidebar_list': all_strats,
               'sidebar_default_asset': default_asset_qs[0]['asset_ticker']}

    return render(request,'interpreting_results.html',context)

def disclaimer(request):
    all_strats_qs = Strategies.objects.all().values('strategy_reference')
    all_strats = [item['strategy_reference'].replace('_', ' ').upper() for item in all_strats_qs]

    default_asset_qs = Assets.objects.filter(pk=1).values('asset_ticker')
    context = {'strats_sidebar_list': all_strats,
               'sidebar_default_asset': default_asset_qs[0]['asset_ticker']}

    return render(request, 'disclaimer.html', context)

def methodology(request):
    all_strats_qs = Strategies.objects.all().values('strategy_reference')
    all_strats = [item['strategy_reference'].replace('_', ' ').upper() for item in all_strats_qs]

    default_asset_qs = Assets.objects.filter(pk=1).values('asset_ticker')
    context = {'strats_sidebar_list': all_strats,
               'sidebar_default_asset': default_asset_qs[0]['asset_ticker']}

    return render(request, 'methodology.html',context)

def about(request):
    all_strats_qs = Strategies.objects.all().values('strategy_reference')
    all_strats = [item['strategy_reference'].replace('_', ' ').upper() for item in all_strats_qs]

    default_asset_qs = Assets.objects.filter(pk=1).values('asset_ticker')
    context = {'strats_sidebar_list': all_strats,
               'sidebar_default_asset': default_asset_qs[0]['asset_ticker']}

    return render(request, 'about.html',context)
