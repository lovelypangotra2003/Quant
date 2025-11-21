# from alpaca_trade_api import REST

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
# 🔑 Replace these with your actual keys from Alpaca (paper or live)
API_KEY = "PKTDKW5HWOUWKZTVOPWDP5QUFJ"
API_SECRET = "59L2sfAV3tradY6udz4okAbCDZ8ZJzXDfYzCi18ZVLpF"
BASE_URL = "https://paper-api.alpaca.markets"

# Connect to paper trading
client = TradingClient(API_KEY, API_SECRET, paper=True)

# Fetch account info
account = client.get_account()
print(f"Account status: {account.status}")
print(f"Buying power: {account.buying_power}")

# Fetch open orders properly
orders_request = GetOrdersRequest(status='open')
orders = client.get_orders(orders_request)

print(f"Open orders: {len(orders)}")
for o in orders:
    print(f"- {o.symbol}: {o.qty} {o.side}")