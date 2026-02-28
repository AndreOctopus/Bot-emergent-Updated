"""
AI Crypto Trading Bot - Binance Futures Scalping
All API keys hardcoded for easy deployment
Target: +2.5% daily profit (Conservative Mode)
"""

import asyncio
import logging
import hmac
import hashlib
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from decimal import Decimal, ROUND_DOWN
from enum import Enum
import aiohttp
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
import os

logger = logging.getLogger(__name__)

# ============== HARDCODED API KEYS ==============
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BINANCE_API_KEY = "10O2JrH2o1YIZc23Gs9CahTAqWcCsteQEpNqYsKbK4DIIOQmPomNofYY38qLe4iW"
BINANCE_SECRET_KEY = "xoFE8rfhm8P2chKd4FNvcKyRtZDJVPymIFvlkTgBiUqnxq3ypS1UdPSFf8JvrbPi"
TELEGRAM_BOT_TOKEN = "8144215710:AAEX_U3V2HQhJ5AhVpFJbt5CRm2dA5yDiHg"
TELEGRAM_CHAT_ID = "-1003844330472"
# ================================================

# ============== Models ==============

class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class TradeSignal(BaseModel):
    symbol: str
    side: TradeSide
    confidence: float
    reason: str
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int = 3
    position_size_percent: float = 3.0

class Position(BaseModel):
    symbol: str
    side: str
    entry_price: float
    quantity: float
    unrealized_pnl: float
    leverage: int
    liquidation_price: float
    mark_price: float

class DailyStats(BaseModel):
    date: str
    starting_balance: float
    current_balance: float
    profit_loss: float
    profit_loss_percent: float
    trades_count: int
    winning_trades: int
    losing_trades: int
    target_reached: bool


# ============== Telegram Notifier ==============

class TelegramNotifier:
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
    
    async def send_message(self, message: str):
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                data = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
                async with session.post(url, json=data) as resp:
                    if resp.status != 200:
                        logger.error(f"Telegram error: {await resp.text()}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
    
    async def send_trade_notification(self, signal: TradeSignal, action: str):
        emoji = "🟢" if signal.side == TradeSide.BUY else "🔴"
        message = f"""
{emoji} <b>{action}: {signal.symbol}</b>

📊 <b>Side:</b> {signal.side.value}
💰 <b>Entry:</b> ${signal.entry_price:,.2f}
🎯 <b>TP:</b> ${signal.take_profit:,.2f}
🛑 <b>SL:</b> ${signal.stop_loss:,.2f}
⚡ <b>Leverage:</b> {signal.leverage}x
📈 <b>Confidence:</b> {signal.confidence:.1f}%

💡 {signal.reason}
"""
        await self.send_message(message)
    
    async def send_daily_report(self, stats: DailyStats):
        status_emoji = "✅" if stats.target_reached else "📊"
        pnl_emoji = "📈" if stats.profit_loss >= 0 else "📉"
        win_rate = (stats.winning_trades / stats.trades_count * 100) if stats.trades_count > 0 else 0
        
        message = f"""
{status_emoji} <b>DAILY REPORT - {stats.date}</b>

💰 Start: ${stats.starting_balance:,.2f}
💵 Current: ${stats.current_balance:,.2f}
{pnl_emoji} P&L: ${stats.profit_loss:,.2f} ({stats.profit_loss_percent:+.2f}%)

📊 Trades: {stats.trades_count} | Win: {stats.winning_trades} | Loss: {stats.losing_trades}
🎯 Win Rate: {win_rate:.1f}%
"""
        await self.send_message(message)
    
    async def send_bot_status(self, status: str, details: str = ""):
        message = f"🤖 <b>BOT: {status}</b>\n\n{details}\n\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        await self.send_message(message)


# ============== Binance Futures Client ==============

class BinanceFuturesClient:
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        self.api_key = api_key or BINANCE_API_KEY
        self.api_secret = api_secret or BINANCE_SECRET_KEY
        self.base_url = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def start(self):
        self.session = aiohttp.ClientSession()
    
    async def stop(self):
        if self.session:
            await self.session.close()
    
    def _sign(self, params: Dict) -> Dict:
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = 5000
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(self.api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        params['signature'] = signature
        return params
    
    async def _request(self, method: str, endpoint: str, signed: bool = False, params: Dict = None) -> Dict:
        if params is None:
            params = {}
        if signed:
            params = self._sign(params)
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        async with self.session.request(method, url, params=params, headers=headers) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise Exception(f"Binance API error: {data}")
            return data
    
    async def get_account(self) -> Dict:
        return await self._request("GET", "/fapi/v2/account", signed=True)
    
    async def get_balance(self) -> float:
        account = await self.get_account()
        for asset in account.get('assets', []):
            if asset['asset'] == 'USDT':
                return float(asset['walletBalance'])
        return 0.0
    
    async def get_positions(self) -> List[Dict]:
        return await self._request("GET", "/fapi/v2/positionRisk", signed=True)
    
    async def get_open_positions(self) -> List[Position]:
        positions = await self.get_positions()
        result = []
        for p in positions:
            if float(p['positionAmt']) != 0:
                result.append(Position(
                    symbol=p['symbol'],
                    side="LONG" if float(p['positionAmt']) > 0 else "SHORT",
                    entry_price=float(p['entryPrice']),
                    quantity=abs(float(p['positionAmt'])),
                    unrealized_pnl=float(p['unRealizedProfit']),
                    leverage=int(p['leverage']),
                    liquidation_price=float(p['liquidationPrice']),
                    mark_price=float(p['markPrice'])
                ))
        return result
    
    async def get_ticker_price(self, symbol: str) -> float:
        data = await self._request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol})
        return float(data['price'])
    
    async def get_klines(self, symbol: str, interval: str = "15m", limit: int = 100) -> List:
        return await self._request("GET", "/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    
    async def get_exchange_info(self, symbol: str) -> Dict:
        data = await self._request("GET", "/fapi/v1/exchangeInfo")
        for s in data['symbols']:
            if s['symbol'] == symbol:
                return s
        return {}
    
    async def set_leverage(self, symbol: str, leverage: int):
        return await self._request("POST", "/fapi/v1/leverage", signed=True, params={"symbol": symbol, "leverage": leverage})
    
    async def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET", reduce_only: bool = False) -> Dict:
        params = {"symbol": symbol, "side": side, "type": order_type, "quantity": quantity}
        if reduce_only:
            params["reduceOnly"] = "true"
        return await self._request("POST", "/fapi/v1/order", signed=True, params=params)
    
    async def close_position(self, symbol: str, side: str, quantity: float):
        close_side = "SELL" if side == "LONG" else "BUY"
        return await self.place_order(symbol=symbol, side=close_side, quantity=quantity, reduce_only=True)


# ============== AI Strategy Engine ==============

class AIStrategyEngine:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.model = "gpt-4o"
    
    async def analyze_market(self, symbol: str, klines: List, current_price: float, balance: float, positions: List[Position]) -> Optional[TradeSignal]:
        closes = [float(k[4]) for k in klines[-50:]]
        highs = [float(k[2]) for k in klines[-50:]]
        lows = [float(k[3]) for k in klines[-50:]]
        volumes = [float(k[5]) for k in klines[-50:]]
        
        sma_10 = sum(closes[-10:]) / 10
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50
        
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            gains.append(diff if diff > 0 else 0)
            losses.append(abs(diff) if diff < 0 else 0)
        
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0.0001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        
        avg_volume = sum(volumes[-20:]) / 20
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
        price_change_1h = ((closes[-1] - closes[-4]) / closes[-4] * 100) if len(closes) >= 4 else 0
        
        market_data = f"""
SYMBOL: {symbol} | PRICE: ${current_price:,.2f} | BALANCE: ${balance:,.2f}
SMA10: ${sma_10:,.2f} | SMA20: ${sma_20:,.2f} | RSI: {rsi:.1f}
Volume: {volume_ratio:.2f}x | 1H Change: {price_change_1h:+.2f}%
Trend: {'BULLISH' if sma_10 > sma_20 > sma_50 else 'BEARISH' if sma_10 < sma_20 < sma_50 else 'RANGING'}
Positions: {len(positions)}
"""
        
        prompt = f"""Analyze this crypto market data and decide: LONG, SHORT, or WAIT.
{market_data}
Rules: Only trade with trend, min 80% confidence, 3x leverage, 1% SL, 1.5% TP.
Respond JSON only: {{"action": "LONG"|"SHORT"|"WAIT", "confidence": 0-100, "reason": "brief"}}"""

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            import re
            response_text = response.choices[0].message.content
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                return None
            
            if decision.get('action') == 'WAIT':
                return None
            
            side = TradeSide.BUY if decision['action'] == 'LONG' else TradeSide.SELL
            sl_pct, tp_pct = 0.01, 0.015
            
            if side == TradeSide.BUY:
                stop_loss = current_price * (1 - sl_pct)
                take_profit = current_price * (1 + tp_pct)
            else:
                stop_loss = current_price * (1 + sl_pct)
                take_profit = current_price * (1 - tp_pct)
            
            return TradeSignal(
                symbol=symbol, side=side, confidence=decision.get('confidence', 50),
                reason=decision.get('reason', 'AI'), entry_price=current_price,
                stop_loss=stop_loss, take_profit=take_profit, leverage=3, position_size_percent=3
            )
        except Exception as e:
            logger.error(f"AI error: {e}")
            return None


# ============== Main Trading Bot ==============

class CryptoTradingBot:
    TRADING_PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    
    def __init__(self, binance_key: str = None, binance_secret: str = None, telegram_token: str = None, 
             telegram_chat_id: str = None, openai_key: str = None, db = None, testnet: bool = True):
        self.binance = BinanceFuturesClient(binance_key, binance_secret, testnet)
        self.telegram = TelegramNotifier(telegram_token, telegram_chat_id)
        self.ai_engine = AIStrategyEngine(openai_key)
        self.db = db
        
        self.daily_target_percent = 2.5
        self.max_positions = 2
        self.leverage = 3
        self.stop_loss_percent = 1.0
        self.take_profit_percent = 1.5
        self.risk_per_trade = 3.0
        self.running = False
        
        self.daily_start_balance = 0.0
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.last_report_date = None
    
    async def start(self):
        await self.binance.start()
        self.running = True
        self.daily_start_balance = await self.binance.get_balance()
        self.last_report_date = datetime.now(timezone.utc).date()
        
        await self.telegram.send_bot_status(
            "STARTED 🚀",
            f"💰 Balance: ${self.daily_start_balance:,.2f}\n🎯 Target: +{self.daily_target_percent}%\n⚡ Leverage: {self.leverage}x"
        )
    
    async def stop(self):
        self.running = False
        await self.binance.stop()
        await self.telegram.send_bot_status("STOPPED 🛑")
    
    async def run_trading_loop(self):
        while self.running:
            try:
                current_balance = await self.binance.get_balance()
                daily_pnl = ((current_balance - self.daily_start_balance) / self.daily_start_balance * 100) if self.daily_start_balance > 0 else 0
                
                if daily_pnl >= self.daily_target_percent:
                    await self.telegram.send_message(f"🎉 TARGET REACHED! +{daily_pnl:.2f}%")
                    await asyncio.sleep(3600)
                    continue
                
                positions = await self.binance.get_open_positions()
                await self._manage_positions(positions)
                
                if len(positions) < self.max_positions:
                    await self._find_opportunities(positions, current_balance)
                
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(60)
    
    async def _find_opportunities(self, positions: List[Position], balance: float):
        current_symbols = {p.symbol for p in positions}
        for symbol in self.TRADING_PAIRS:
            if symbol in current_symbols or not self.running:
                continue
            try:
                price = await self.binance.get_ticker_price(symbol)
                klines = await self.binance.get_klines(symbol, "15m", 100)
                signal = await self.ai_engine.analyze_market(symbol, klines, price, balance, positions)
                
                if signal and signal.confidence >= 80:
                    await self._execute_trade(signal, balance)
                    return
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Error {symbol}: {e}")
    
    async def _execute_trade(self, signal: TradeSignal, balance: float):
        try:
            await self.binance.set_leverage(signal.symbol, self.leverage)
            position_value = balance * (self.risk_per_trade / 100) * self.leverage
            quantity = position_value / signal.entry_price
            
            info = await self.binance.get_exchange_info(signal.symbol)
            precision = 3
            for f in info.get('filters', []):
                if f['filterType'] == 'LOT_SIZE':
                    step = float(f['stepSize'])
                    precision = len(str(step).rstrip('0').split('.')[-1]) if '.' in str(step) else 0
            
            quantity = float(Decimal(str(quantity)).quantize(Decimal(10) ** -precision, rounding=ROUND_DOWN))
            if quantity <= 0:
                return
            
            await self.binance.place_order(signal.symbol, signal.side.value, quantity)
            await self.telegram.send_trade_notification(signal, "OPENED")
            self.trades_today += 1
            
            if self.db:
                await self.db.trades.insert_one({
                    "symbol": signal.symbol, "side": signal.side.value, "quantity": quantity,
                    "entry_price": signal.entry_price, "timestamp": datetime.now(timezone.utc)
                })
        except Exception as e:
            logger.error(f"Trade error: {e}")
            await self.telegram.send_message(f"⚠️ Trade failed: {str(e)[:100]}")
    
    async def _manage_positions(self, positions: List[Position]):
        for pos in positions:
            try:
                entry_value = pos.entry_price * pos.quantity
                pnl_pct = (pos.unrealized_pnl / entry_value * 100) if entry_value > 0 else 0
                
                tp_threshold = self.take_profit_percent * self.leverage
                sl_threshold = -self.stop_loss_percent * self.leverage
                
                if pnl_pct >= tp_threshold:
                    await self.binance.close_position(pos.symbol, pos.side, pos.quantity)
                    await self.telegram.send_message(f"✅ CLOSED {pos.symbol}: ${pos.unrealized_pnl:+.2f} (TP)")
                    self.winning_trades += 1
                elif pnl_pct <= sl_threshold:
                    await self.binance.close_position(pos.symbol, pos.side, pos.quantity)
                    await self.telegram.send_message(f"❌ CLOSED {pos.symbol}: ${pos.unrealized_pnl:+.2f} (SL)")
                    self.losing_trades += 1
            except Exception as e:
                logger.error(f"Position error: {e}")


# ============== Bot Manager ==============

class TradingBotManager:
    _instance: Optional[CryptoTradingBot] = None
    _task: Optional[asyncio.Task] = None
    
    @classmethod
    async def start_bot(cls, binance_key: str = None, binance_secret: str = None, telegram_token: str = None,
                        telegram_chat_id: str = None, openai_key: str = None, db = None) -> bool:
        if cls._instance and cls._instance.running:
            return False
        
        cls._instance = CryptoTradingBot(
            binance_key=binance_key or BINANCE_API_KEY,
            binance_secret=binance_secret or BINANCE_SECRET_KEY,
            telegram_token=telegram_token or TELEGRAM_BOT_TOKEN,
            telegram_chat_id=telegram_chat_id or TELEGRAM_CHAT_ID,
            openai_key=openai_key or OPENAI_API_KEY,
            db=db
        )
        
        await cls._instance.start()
        cls._task = asyncio.create_task(cls._instance.run_trading_loop())
        return True
    
    @classmethod
    async def stop_bot(cls) -> bool:
        if not cls._instance:
            return False
        await cls._instance.stop()
        if cls._task:
            cls._task.cancel()
        cls._instance = None
        cls._task = None
        return True
    
    @classmethod
    def get_status(cls) -> Dict:
        if not cls._instance:
            return {"running": False}
        return {
            "running": cls._instance.running,
            "daily_start_balance": cls._instance.daily_start_balance,
            "trades_today": cls._instance.trades_today,
            "winning_trades": cls._instance.winning_trades,
            "losing_trades": cls._instance.losing_trades
        }
