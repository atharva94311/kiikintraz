from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import requests

app = Flask(__name__)

# ── Load model once at startup ──────────────────────────────────────
model = joblib.load("crypto_model.pkl")

# ── Yahoo Finance config ──────────────────────────────────────────
# Map display names → Yahoo Finance Tickers
COINS = {
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Litecoin": "LTC-USD",
    "XRP": "XRP-USD", "Dogecoin": "DOGE-USD", "Cardano": "ADA-USD",
    "Solana": "SOL-USD", "Polkadot": "DOT-USD", "BinanceCoin": "BNB-USD",
    "ChainLink": "LINK-USD", "Uniswap": "UNI-USD", "Aave": "AAVE-USD",
    "Cosmos": "ATOM-USD", "EOS": "EOS-USD", "Tron": "TRX-USD", "Stellar": "XLM-USD",
    "Monero": "XMR-USD", "Iota": "IOTA-USD", "NEM": "XEM-USD", "Tether": "USDT-USD",
    "USDCoin": "USDC-USD", "CryptocomCoin": "CRO-USD", "WrappedBitcoin": "WBTC-USD",
}

FEATURES = [
    "volatility_7", "Close", "Volume", "vol_relative",
    "z_score_7", "RSI_14", "return_1d", "momentum_3",
    "acceleration", "price_vs_ma7",
]

# ── Feature engineering (mirrors your notebook) ────────────────────
def compute_features(prices, volumes):
    """Return a single feature vector from ≥60 days of price+volume data."""
    prices = np.array(prices, dtype=float)
    volumes = np.array(volumes, dtype=float)

    close = prices[-1]

    # return_1d
    return_1d = (prices[-1] - prices[-2]) / prices[-2]

    # momentum_3
    momentum_3 = (prices[-1] - prices[-4]) / prices[-4]

    # ma_7, ma_14
    ma_7 = prices[-7:].mean()
    ma_14 = prices[-14:].mean()

    # price_vs_ma7
    price_vs_ma7 = close / ma_7

    # volatility_7  (std of daily returns over last 7 days)
    returns_8 = np.diff(prices[-8:]) / prices[-8:-1]  # 7 daily returns
    volatility_7 = returns_8.std()

    # vol_30  (std of closing prices over last 30 days)
    vol_30 = prices[-30:].std()
    vol_relative = volatility_7 / vol_30 if vol_30 != 0 else 0

    # z_score_7
    z_score_7 = (close - ma_7) / volatility_7 if volatility_7 != 0 else 0

    # RSI 14
    deltas = np.diff(prices[-15:])  # 14 deltas
    gains = np.where(deltas > 0, deltas, 0).mean()
    losses = np.where(deltas < 0, -deltas, 0).mean()
    rs = gains / losses if losses != 0 else 100
    rsi_14 = 100 - (100 / (1 + rs))

    # acceleration
    prev_return = (prices[-2] - prices[-3]) / prices[-3]
    acceleration = return_1d - prev_return

    # volume (latest day)
    volume = volumes[-1]

    # vol_spike (not in feature list but kept for safety)
    # using vol_relative instead

    return {
        "volatility_7": volatility_7,
        "Close": close,
        "Volume": volume,
        "vol_relative": vol_relative,
        "z_score_7": z_score_7,
        "RSI_14": rsi_14,
        "return_1d": return_1d,
        "momentum_3": momentum_3,
        "acceleration": acceleration,
        "price_vs_ma7": price_vs_ma7,
    }


def fetch_yahoo(ticker):
    """Fetch 60 days of daily prices + volumes from Yahoo Finance."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": "1d", "range": "60d"}
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    result = data["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    
    raw_prices = quote["close"]
    raw_volumes = quote["volume"]
    
    # Filter out potential None values (Yahoo sometimes has nulls)
    prices, volumes = [], []
    for p, v in zip(raw_prices, raw_volumes):
        if p is not None and v is not None:
            prices.append(p)
            volumes.append(v)
            
    return prices[-60:], volumes[-60:]


# ── Routes ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", coins=COINS)


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    coin_name = body.get("coin", "")
    ticker = COINS.get(coin_name)

    if not ticker:
        return jsonify({"error": f"Unknown coin: {coin_name}"}), 400

    try:
        prices, volumes = fetch_yahoo(ticker)
        if len(prices) < 30:
            return jsonify({"error": f"Not enough data for {coin_name} from Yahoo Finance."}), 500
            
        feats = compute_features(prices, volumes)
        X = np.array([[feats[f] for f in FEATURES]])
        pred = model.predict(X)[0]
        label = "UP" if pred == 1 else "DOWN"
        
        return jsonify({"prediction": label, "coin": coin_name, "features": feats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
