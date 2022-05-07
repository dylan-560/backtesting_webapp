import mysql.connector
import pandas as pd
import numpy
import os
from datetime import datetime, timedelta
import json
"""
git add website/data
git commit -am 'update message'
git push

heroku login
git push heroku main
"""


root_data_dir = os.path.join('website','data')
root_dir_list = ['asset_info', 'strategies_info', 'candle_data', 'eval_metrics']

def update_local_data_files():

    def delete_files():
        print('deleting previous files...')
        # remove all files
        for root, dirs, files in os.walk(root_data_dir):
            for file in files:
                print('     ...', os.path.join(root, file))
                os.remove(os.path.join(root, file))

        # remove strat dirs in eval metrics file
        if os.path.exists(os.path.join(root_data_dir, 'eval_metrics')):
            for strat_file in os.listdir(os.path.join(root_data_dir, 'eval_metrics')):
                for asset_file in os.listdir(os.path.join(root_data_dir, 'eval_metrics', strat_file)):
                    for results_file in os.listdir(os.path.join(root_data_dir, 'eval_metrics', strat_file, asset_file)):
                        # print('             ...',os.path.join(root_data_dir,'eval_metrics',strat_file,asset_file,results_file))
                        os.remove(os.path.join(root_data_dir, 'eval_metrics', strat_file, asset_file, results_file))

                    # print('         ...',os.path.join(root_data_dir,'eval_metrics',strat_file,asset_file))
                    os.rmdir(os.path.join(root_data_dir, 'eval_metrics', strat_file, asset_file))

                # print('     ...',os.path.join(root_data_dir,'eval_metrics',strat_file))
                os.rmdir(os.path.join(root_data_dir, 'eval_metrics', strat_file))

    def recreate_data_directories(strats_list, assets_list):
        # create the root if it doesnt exist
        if not os.path.exists(root_data_dir):
            os.mkdir(root_data_dir)

        # create dirs
        for d in root_dir_list:
            dir_path = os.path.join(root_data_dir, d)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            # prepopulate the eval metrics directory
            if 'eval_metrics' in dir_path:
                for strat in strats_list:

                    strat_ref = strat['strategy_reference'].lower()

                    if not os.path.exists(os.path.join(dir_path, strat_ref)):
                        os.mkdir(os.path.join(dir_path, strat_ref))

                    for asset in assets_list:
                        if not os.path.exists(
                                os.path.join(dir_path, strat_ref, asset['asset_ticker'])):
                            os.mkdir(os.path.join(dir_path, strat_ref, asset['asset_ticker']))

    def establish_DB_conection():
        mydb = mysql.connector.connect(host='localhost',
                                       user='root',
                                       passwd='afroman3',
                                       database='django_backtesting_db',
                                       auth_plugin='mysql_native_password')

        return mydb

    def pull_all_assets_info_from_DB():
        query = "SELECT * FROM assets;"

        cursor = db_obj.cursor(dictionary=True)

        cursor.execute(query)
        q_results = cursor.fetchall()

        ret_val = []
        for a in q_results:
            ret_val.append(a)

        cursor.close()

        return ret_val

    def pull_all_strategies_info_from_DB():
        query = "SELECT * FROM strategies;"

        cursor = db_obj.cursor(dictionary=True)

        cursor.execute(query)
        q_results = cursor.fetchall()

        ret_val = []
        for a in q_results:
            ret_val.append(a)

        cursor.close()

        return ret_val

    def pull_all_candle_data(assets_list):

        for asset in assets_list:
            query = "SELECT datetime, open, high, low, close, volume " \
                    "FROM asset_ohlcv " \
                    "WHERE asset_ticker='" + asset['asset_ticker'] + "';"

            data_df = pd.read_sql(query, db_obj)
            if data_df.empty:
                print(asset['asset_ticker'], '- no data found...')
                continue

            try:
                data_df.drop(data_df.columns.difference(['datetime', 'open', 'high', 'low', 'close', 'volume']), 1,
                             inplace=True)
                data_df.to_csv(os.path.join(root_data_dir, 'candle_data', asset['asset_ticker'] + '.csv'), index=False)
            except Exception as E:
                print('ERROR:', E)
            print()

    def pull_all_top_x_eval_results(strats_info, assets_info):

        def save_results(strat_info, asset_info, orient, start_date, end_date, label):

            base_query = "SELECT * " \
                         "FROM evaluation_metrics " \
                         "WHERE period_end_date='" + end_date + "' AND " \
                                                                "period_start_date='" + start_date + "' AND " \
                                                                                                     "strat_id=" + str(
                strat_info['strategy_id']) + " AND " \
                                             "asset_id=" + str(asset_info['asset_id']) + " " \
                                                                                         "ORDER BY total_realized_R " + orient + " limit 5;"

            df = pd.read_sql(base_query, db_obj)
            # if df.empty:
            #     #print(label,strat_info['strategy_reference'],asset_info['asset_ticker'],'- no data found')
            #     return

            count_query = "SELECT COUNT(*) " \
                          "FROM evaluation_metrics " \
                          "WHERE period_end_date='" + end_date + "' AND " \
                                                                 "period_start_date='" + start_date + "' AND " \
                                                                                                      "strat_id=" + str(
                strat_info['strategy_id']) + " AND " \
                                             "asset_id=" + str(asset_info['asset_id']) + ";"

            cursor.execute(count_query)
            total_count = cursor.fetchall()[0][0]

            df['total_count'] = total_count

            df['strategy_params'] = df['strategy_params'].apply(json.loads)
            df['trades_list'] = df['trades_list'].apply(json.loads)

            strat_ref = strat_info['strategy_reference'].lower()

            if orient == 'desc':
                filename = strat_ref + '_' + asset_info[
                    'asset_ticker'] + '_' + label + '_' + 'topresults.csv'
            if orient == 'asc':
                filename = strat_ref + '_' + asset_info[
                    'asset_ticker'] + '_' + label + '_' + 'bottomresults.csv'

            save_filepath = os.path.join(root_data_dir,
                                         'eval_metrics',
                                         strat_ref,
                                         asset_info['asset_ticker'],
                                         filename)

            df.to_csv(save_filepath, index=False)

        #############################################

        month_0_end = datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_0_start = (month_0_end - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_1_end = month_0_start
        month_1_start = (month_1_end - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        db_obj = establish_DB_conection()
        cursor = db_obj.cursor()

        for strat in strats_info:
            for asset in assets_info:
                # previous month best and worst
                save_results(strat_info=strat,
                             asset_info=asset,
                             orient='desc',
                             start_date=str(month_1_start).split(' ')[0],
                             end_date=str(month_1_end).split(' ')[0],
                             label='prevmonth')

                save_results(strat_info=strat,
                             asset_info=asset,
                             orient='asc',
                             start_date=str(month_1_start).split(' ')[0],
                             end_date=str(month_1_end).split(' ')[0],
                             label='prevmonth')

                # current month best and worst
                save_results(strat_info=strat,
                             asset_info=asset,
                             orient='desc',
                             start_date=str(month_0_start).split(' ')[0],
                             end_date=str(month_0_end).split(' ')[0],
                             label='currmonth')

                save_results(strat_info=strat,
                             asset_info=asset,
                             orient='asc',
                             start_date=str(month_0_start).split(' ')[0],
                             end_date=str(month_0_end).split(' ')[0],
                             label='currmonth')

        db_obj.close()

    ##################################################################
    ##################################################################
    ##################################################################

    delete_files()

    db_obj = establish_DB_conection()

    # pull strats and assets info
    all_strats_info = pull_all_strategies_info_from_DB()
    all_assets_info = pull_all_assets_info_from_DB()

    recreate_data_directories(strats_list=all_strats_info, assets_list=all_assets_info)

    # save strats and assets info as csv
    pd.DataFrame(all_assets_info).to_csv(os.path.join(root_data_dir, 'asset_info', 'all_asset_info.csv'), index=False)
    pd.DataFrame(all_strats_info).to_csv(os.path.join(root_data_dir, 'strategies_info', 'all_strats_info.csv'),
                                         index=False)

    pull_all_candle_data(assets_list=all_assets_info)

    pull_all_top_x_eval_results(strats_info=all_strats_info, assets_info=all_assets_info)

# update_local_data_files()