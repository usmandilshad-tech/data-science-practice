## By Muhammad Usman Dilshad, usmandilshad@gmail.com

import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
from prophet import Prophet

# For ARIMA model
from pmdarima import auto_arima
# Used Numpy 1.26.3 version as latest one was giving an error. Installed all packages and in the last install PMDarima to ensure it has all the packages.

###############################################################################
# 1. Data Retrieval
###############################################################################

def get_coin_market_data(coin_id, vs_currency="usd", days=365):
    """
    Fetches historical day-wise market data for a given coin from CoinGecko
    over a specified number of days (default: 365 days).
    """
    end_time = int(time.time())  # current UNIX time (seconds)
    start_time = end_time - days * 24 * 60 * 60

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"

    params = {
        'vs_currency': vs_currency,
        'from': start_time,
        'to': end_time
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "prices" not in data:
        raise ValueError(f"Data not returned properly for {coin_id}. Response: {data}")

    # Convert the 'prices' part of the data to a Pandas DataFrame
    df_prices = pd.DataFrame(data['prices'], columns=['ds', 'y'])
    # Convert ds (ms) to datetime
    df_prices['ds'] = pd.to_datetime(df_prices['ds'], unit='ms')
    # Return a DataFrame with columns [ds, y]
    return df_prices


###############################################################################
# 2. Modeling with Prophet
###############################################################################

def forecast_with_prophet(df, periods=7):
    """
    Given a DataFrame with columns [ds, y], train a Prophet model and forecast.
    Returns the forecast DataFrame (which contains ds, yhat, yhat_lower, yhat_upper)
    and the fitted model.
    """
    prophet_model = Prophet()
    prophet_model.fit(df)
    future = prophet_model.make_future_dataframe(periods=periods)
    forecast = prophet_model.predict(future)
    return forecast, prophet_model


###############################################################################
# 3. Modeling with ARIMA (using pmdarima.auto_arima)
###############################################################################

def forecast_with_arima(df, periods=7):
    """
    Given a DataFrame with columns [ds, y], train an ARIMA model via pmdarima's auto_arima.
    Returns a Pandas Series with forecasts for the requested period.
    """
    # ARIMA typically wants a univariate series indexed by time
    # We'll set 'ds' as the index, and use 'y' as the target.
    df_arima = df.set_index('ds').copy()
    df_arima = df_arima.asfreq('D')  # ensure daily frequency

    # auto_arima tries to find the best p,d,q automatically
    arima_model = auto_arima(
        df_arima['y'],
        seasonal=False,
        trace=False,
        error_action='ignore',
        suppress_warnings=True
    )
    # Fit final model
    arima_model.fit(df_arima['y'])

    # Forecast
    forecast_arima = arima_model.predict(n_periods=periods)
    # Convert to a Pandas Series with a daily date index continuing past the last known date
    last_date = df_arima.index[-1]
    date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    forecast_series = pd.Series(forecast_arima, index=date_range, name='yhat_arima')
    return forecast_series


###############################################################################
# 4. Main: Compare Forecasts for Multiple Coins
###############################################################################

if __name__ == "__main__":
    coins = ["bitcoin", "ripple", "dogecoin", "ethereum", "floki"]
    vs_currency = "usd"
    days_to_predict = 7

    # Dictionary to store results for each coin
    results = {}

    plt.figure(figsize=(12, 10))  # We'll create subplots for all coins

    for idx, coin_id in enumerate(coins, start=1):
        # 4.1. Get data
        df = get_coin_market_data(coin_id, vs_currency, days=365)

        # 4.2. Prophet Forecast
        forecast_prophet, prophet_model = forecast_with_prophet(df, periods=days_to_predict)

        # 4.3. ARIMA Forecast
        forecast_arima = forecast_with_arima(df, periods=days_to_predict)

        # 4.4. Store final 7 days from Prophet in a DataFrame
        prophet_future = forecast_prophet[['ds', 'yhat']].tail(days_to_predict).copy()
        prophet_future.set_index('ds', inplace=True)

        # 4.5. Combine both forecasts in a table
        compare_df = prophet_future.join(forecast_arima, how='outer')  # 'yhat' & 'yhat_arima'
        compare_df.columns = ['Prophet', 'ARIMA']

        # 4.6. Store the table for later printing
        results[coin_id] = compare_df

        # 4.7. Plot: We'll do a small subplot for each coin
        plt.subplot(3, 2, idx)  # 5 coins => up to (3 rows x 2 cols)

        # Plot historical data (last 365 days)
        plt.plot(df['ds'], df['y'], 'k.-', label='Historical')

        # Plot Prophet forecast
        plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], 'b-', label='Prophet Forecast')

        # Plot ARIMA forecast
        plt.plot(compare_df.index, compare_df['ARIMA'], 'r--', label='ARIMA Forecast')

        plt.title(f"{coin_id.capitalize()} Forecast")
        plt.xlabel("Date")
        plt.ylabel(f"Price ({vs_currency.upper()})")
        plt.legend()
        plt.tight_layout()

    plt.suptitle("Crypto Price Forecast Comparison (Prophet vs ARIMA)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    ###############################################################################
    # 5. Print tables of final 7-day forecasts for each coin
    ###############################################################################
    for coin_id in coins:
        print(f"\n=== 7-Day Forecast Comparison for {coin_id.capitalize()} ===")
        print(results[coin_id])
