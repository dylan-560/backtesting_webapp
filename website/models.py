from django.db import models

# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.


class AssetsOHLCV(models.Model):
    candle_id = models.AutoField(primary_key=True)
    asset_ticker_identification = models.IntegerField()
    asset_ticker = models.CharField(db_column='asset_ticker', max_length=9, blank=True, null=True)  # Field renamed because of name conflict.
    timeframe = models.CharField(max_length=5)
    datetime = models.DateTimeField()
    open = models.DecimalField(max_digits=12, decimal_places=5, blank=True, null=True)
    high = models.DecimalField(max_digits=12, decimal_places=5, blank=True, null=True)
    low = models.DecimalField(max_digits=12, decimal_places=5, blank=True, null=True)
    close = models.DecimalField(max_digits=12, decimal_places=5, blank=True, null=True)
    volume = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'asset_ohlcv'

class Assets(models.Model):
    asset_id = models.AutoField(primary_key=True)
    asset_ticker = models.CharField(max_length=10)
    asset_name = models.CharField(max_length=20, blank=True, null=True)
    market = models.CharField(max_length=20)

    class Meta:
        managed = False
        db_table = 'assets'

class EvaluationMetrics(models.Model):
    strat_parameters_id = models.AutoField(primary_key=True)
    strat = models.ForeignKey('Strategies', models.DO_NOTHING)
    asset = models.ForeignKey(Assets, models.DO_NOTHING)
    period_end_date = models.DateTimeField()
    period_start_date = models.DateTimeField()
    timeframe = models.CharField(max_length=5)
    strategy_params = models.CharField(max_length=300)
    total_realized_r = models.DecimalField(db_column='total_realized_R', max_digits=6, decimal_places=3, blank=True, null=True)  # Field name made lowercase.
    strike_rate = models.DecimalField(max_digits=5, decimal_places=4, blank=True, null=True)
    num_trades = models.IntegerField(blank=True, null=True)
    avg_hold_time = models.IntegerField(blank=True, null=True)
    expectancy = models.DecimalField(max_digits=6, decimal_places=3, blank=True, null=True)
    avg_winner = models.DecimalField(max_digits=6, decimal_places=3, blank=True, null=True)
    avg_loser = models.DecimalField(max_digits=6, decimal_places=3, blank=True, null=True)
    largest_winner = models.DecimalField(max_digits=6, decimal_places=3, blank=True, null=True)
    largest_loser = models.DecimalField(max_digits=6, decimal_places=3, blank=True, null=True)
    max_drawdown = models.IntegerField(blank=True, null=True)
    max_drawup = models.IntegerField(blank=True, null=True)
    winners_avg_mae = models.DecimalField(db_column='winners_avg_MAE', max_digits=6, decimal_places=3, blank=True, null=True)  # Field name made lowercase.
    winners_avg_mfe = models.DecimalField(db_column='winners_avg_MFE', max_digits=6, decimal_places=3, blank=True, null=True)  # Field name made lowercase.
    losers_avg_mfe = models.DecimalField(db_column='losers_avg_MFE', max_digits=6, decimal_places=3, blank=True, null=True)  # Field name made lowercase.
    losers_avg_mae = models.DecimalField(db_column='losers_avg_MAE', max_digits=6, decimal_places=3, blank=True, null=True)  # Field name made lowercase.
    winners_std_dev_mae = models.DecimalField(db_column='winners_std_dev_MAE', max_digits=6, decimal_places=3, blank=True, null=True)  # Field name made lowercase.
    winners_std_dev_mfe = models.DecimalField(db_column='winners_std_dev_MFE', max_digits=6, decimal_places=3, blank=True, null=True)  # Field name made lowercase.
    losers_std_dev_mae = models.DecimalField(db_column='losers_std_dev_MAE', max_digits=6, decimal_places=3, blank=True, null=True)  # Field name made lowercase.
    losers_std_dev_mfe = models.DecimalField(db_column='losers_std_dev_MFE', max_digits=6, decimal_places=3, blank=True, null=True)  # Field name made lowercase.
    number_of_bars = models.IntegerField(blank=True, null=True)
    returns_to_asset_corr = models.DecimalField(max_digits=4, decimal_places=3, blank=True, null=True)
    equity_curve_regression_slope = models.DecimalField(max_digits=5, decimal_places=3, blank=True, null=True)
    equity_curve_regression_std_error = models.DecimalField(max_digits=5, decimal_places=3, blank=True, null=True)
    kelly_criterion = models.DecimalField(max_digits=5, decimal_places=3, blank=True, null=True)
    trades_list = models.JSONField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'evaluation_metrics'

class Strategies(models.Model):
    strategy_id = models.AutoField(primary_key=True)
    strategy_reference = models.CharField(max_length=40)
    strategy_name = models.CharField(max_length=40, blank=True, null=True)
    strategy_description = models.CharField(max_length=1000, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'strategies'
