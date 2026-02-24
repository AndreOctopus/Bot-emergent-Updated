"""
AI Crypto Trading Bot - Binance Futures Scalping
Autonomous decision making with risk management
Target: +10% daily profit
"""

import asyncio
import logging
import os
import hmac
import hashlib
import time
import json
import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal, ROUND_DOWN
from enum import Enum
import aiohttp
import httpx
from binance.client import Client
from binance.exceptions import BinanceAPIException
from telegram import Bot
from telegram.constants import ParseMode
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from emergentintegrations.llm.chat import LlmChat, UserMessage

logger = logging.getLogger(__name__)

# ============== Models ==============

class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"

class TradeSignal(BaseModel):
    symbol: str
    side: TradeSide
    confidence: float  # 0-100
    reason: str
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int = 10
    position_size_percent: float = 5.0  # % of balance

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
    """Send trading notifications to Telegram"""
    
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
    
    async def send_message(self, message: str):
        """Send message to Telegram channel"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
    
    async def send_trade_notification(self, signal: TradeSignal, action: str):
        """Send trade entry/exit notification"""
        emoji = "🟢" if signal.side == TradeSide.BUY else "🔴"
        message = f"""
{emoji} <b>{action}: {signal.symbol}</b>

📊 <b>Side:</b> {signal.side.value}
💰 <b>Entry:</b> ${signal.entry_price:,.2f}
🎯 <b>Take Profit:</b> ${signal.take_profit:,.2f}
🛑 <b>Stop Loss:</b> ${signal.stop_loss:,.2f}
⚡ <b>Leverage:</b> {signal.leverage}x
📈 <b>Confidence:</b> {signal.confidence:.1f}%

💡 <b>Reason:</b> {signal.reason}

🕐 {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)
    
    async def send_daily_report(self, stats: DailyStats):
        """Send daily performance report"""
        status_emoji = "✅" if stats.target_reached else "📊"
        pnl_emoji = "📈" if stats.profit_loss >= 0 else "📉"
        
        win_rate = (stats.winning_trades / stats.trades_count * 100) if stats.trades_count > 0 else 0
        
        message = f"""
{status_emoji} <b>DAILY REPORT - {stats.date}</b>

💰 <b>Starting Balance:</b> ${stats.starting_balance:,.2f}
💵 <b>Current Balance:</b> ${stats.current_balance:,.2f}

{pnl_emoji} <b>P&L:</b> ${stats.profit_loss:,.2f} ({stats.profit_loss_percent:+.2f}%)

📊 <b>Trading Stats:</b>
• Total Trades: {stats.trades_count}
• Winning: {stats.winning_trades} ✅
• Losing: {stats.losing_trades} ❌
• Win Rate: {win_rate:.1f}%

🎯 <b>Daily Target (10%):</b> {'REACHED ✅' if stats.target_reached else 'In Progress...'}
"""
        await self.send_message(message)
    
    async def send_bot_status(self, status: str, details: str = ""):
        """Send bot status update"""
        message = f"""
🤖 <b>BOT STATUS: {status}</b>

{details}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await self.send_message(message)


# ============== Binance Futures Client ==============

class BinanceFuturesClient:
    """Async Binance Futures API Client"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_url = "wss://stream.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com"
        
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def start(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def stop(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    def _sign(self, params: Dict) -> Dict:
        """Sign request with HMAC SHA256"""
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = 5000
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        return params
    
    async def _request(self, method: str, endpoint: str, signed: bool = False, params: Dict = None) -> Dict:
        """Make API request"""
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
        """Get futures account info"""
        return await self._request("GET", "/fapi/v2/account", signed=True)
    
    async def get_balance(self) -> float:
        """Get USDT balance"""
        account = await self.get_account()
        for asset in account.get('assets', []):
            if asset['asset'] == 'USDT':
                return float(asset['walletBalance'])
        return 0.0
    
    async def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        return await self._request("GET", "/fapi/v2/positionRisk", signed=True)
    
    async def get_open_positions(self) -> List[Position]:
        """Get non-zero positions"""
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
        """Get current price for symbol"""
        data = await self._request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol})
        return float(data['price'])
    
    async def get_klines(self, symbol: str, interval: str = "15m", limit: int = 100) -> List:
        """Get candlestick data"""
        return await self._request("GET", "/fapi/v1/klines", params={
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        })
    
    async def get_exchange_info(self, symbol: str) -> Dict:
        """Get symbol info for precision"""
        data = await self._request("GET", "/fapi/v1/exchangeInfo")
        for s in data['symbols']:
            if s['symbol'] == symbol:
                return s
        return {}
    
    async def set_leverage(self, symbol: str, leverage: int):
        """Set leverage for symbol"""
        return await self._request("POST", "/fapi/v1/leverage", signed=True, params={
            "symbol": symbol,
            "leverage": leverage
        })
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float = None,
        stop_price: float = None,
        reduce_only: bool = False
    ) -> Dict:
        """Place futures order"""
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity
        }
        
        if price:
            params["price"] = price
            params["timeInForce"] = "GTC"
        
        if stop_price:
            params["stopPrice"] = stop_price
        
        if reduce_only:
            params["reduceOnly"] = "true"
        
        return await self._request("POST", "/fapi/v1/order", signed=True, params=params)
    
    async def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for symbol"""
        return await self._request("DELETE", "/fapi/v1/allOpenOrders", signed=True, params={
            "symbol": symbol
        })
    
    async def close_position(self, symbol: str, side: str, quantity: float):
        """Close a position"""
        close_side = "SELL" if side == "LONG" else "BUY"
        return await self.place_order(
            symbol=symbol,
            side=close_side,
            quantity=quantity,
            reduce_only=True
        )


# ============== AI Strategy Engine ==============

class AIStrategyEngine:
    """AI-powered trading strategy with LLM decision making"""
    
    def __init__(self, llm_key: str):
        self.llm_key = llm_key
        self.chat = LlmChat(api_key=llm_key, model="gpt-5.2")
    
    async def analyze_market(
        self,
        symbol: str,
        klines: List,
        current_price: float,
        balance: float,
        positions: List[Position]
    ) -> Optional[TradeSignal]:
        """Use AI to analyze market and generate trading signal"""
        
        # Calculate technical indicators from klines
        closes = [float(k[4]) for k in klines[-50:]]
        highs = [float(k[2]) for k in klines[-50:]]
        lows = [float(k[3]) for k in klines[-50:]]
        volumes = [float(k[5]) for k in klines[-50:]]
        
        # Simple indicators
        sma_10 = sum(closes[-10:]) / 10
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50
        
        # RSI calculation
        gains = []
        losses = []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))
        
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Volatility (ATR approximation)
        atr_values = []
        for i in range(1, min(14, len(highs))):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            atr_values.append(tr)
        atr = sum(atr_values) / len(atr_values) if atr_values else 0
        
        # Volume analysis
        avg_volume = sum(volumes[-20:]) / 20
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Price momentum
        price_change_1h = ((closes[-1] - closes[-4]) / closes[-4] * 100) if len(closes) >= 4 else 0
        price_change_4h = ((closes[-1] - closes[-16]) / closes[-16] * 100) if len(closes) >= 16 else 0
        
        # Prepare market data for AI
        market_data = f"""
SYMBOL: {symbol}
CURRENT PRICE: ${current_price:,.2f}
ACCOUNT BALANCE: ${balance:,.2f}

TECHNICAL INDICATORS:
- SMA 10: ${sma_10:,.2f} (Price {'above' if current_price > sma_10 else 'below'})
- SMA 20: ${sma_20:,.2f} (Price {'above' if current_price > sma_20 else 'below'})
- SMA 50: ${sma_50:,.2f} (Price {'above' if current_price > sma_50 else 'below'})
- RSI (14): {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
- ATR: ${atr:,.2f} ({atr/current_price*100:.2f}% volatility)
- Volume Ratio: {volume_ratio:.2f}x average

PRICE MOMENTUM:
- 1H Change: {price_change_1h:+.2f}%
- 4H Change: {price_change_4h:+.2f}%

CURRENT POSITIONS:
{self._format_positions(positions)}

MARKET CONTEXT:
- Trend: {'BULLISH' if sma_10 > sma_20 > sma_50 else 'BEARISH' if sma_10 < sma_20 < sma_50 else 'RANGING'}
- Volume: {'HIGH' if volume_ratio > 1.5 else 'LOW' if volume_ratio < 0.5 else 'NORMAL'}
"""
        
        prompt = f"""You are an expert cryptocurrency scalping trader. Analyze this market data and decide whether to open a LONG or SHORT position.

{market_data}

TRADING RULES:
1. Target 0.5-2% profit per trade (scalping)
2. Maximum leverage: 20x (recommend 5-15x based on volatility)
3. Stop loss: Always set 1-3% from entry
4. Risk per trade: 2-10% of balance
5. Don't trade if no clear signal
6. Consider existing positions

Respond in JSON format ONLY:
{{
    "action": "LONG" | "SHORT" | "WAIT",
    "confidence": 0-100,
    "leverage": 5-20,
    "position_size_percent": 2-10,
    "stop_loss_percent": 1-3,
    "take_profit_percent": 0.5-2,
    "reason": "brief explanation"
}}"""

        try:
            response = await asyncio.to_thread(
                self.chat.send_message,
                UserMessage(text=prompt)
            )
            
            # Parse JSON from response
            response_text = response.response if hasattr(response, 'response') else str(response)
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                logger.warning(f"No JSON found in AI response: {response_text}")
                return None
            
            if decision.get('action') == 'WAIT':
                logger.info(f"AI decided to WAIT: {decision.get('reason')}")
                return None
            
            # Calculate actual prices
            side = TradeSide.BUY if decision['action'] == 'LONG' else TradeSide.SELL
            sl_percent = decision.get('stop_loss_percent', 2) / 100
            tp_percent = decision.get('take_profit_percent', 1) / 100
            
            if side == TradeSide.BUY:
                stop_loss = current_price * (1 - sl_percent)
                take_profit = current_price * (1 + tp_percent)
            else:
                stop_loss = current_price * (1 + sl_percent)
                take_profit = current_price * (1 - tp_percent)
            
            return TradeSignal(
                symbol=symbol,
                side=side,
                confidence=decision.get('confidence', 50),
                reason=decision.get('reason', 'AI Analysis'),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=decision.get('leverage', 10),
                position_size_percent=decision.get('position_size_percent', 5)
            )
            
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return None
    
    def _format_positions(self, positions: List[Position]) -> str:
        if not positions:
            return "No open positions"
        
        lines = []
        for p in positions:
            pnl_str = f"+${p.unrealized_pnl:.2f}" if p.unrealized_pnl >= 0 else f"-${abs(p.unrealized_pnl):.2f}"
            lines.append(f"- {p.symbol}: {p.side} {p.quantity} @ ${p.entry_price:.2f} | PnL: {pnl_str}")
        return "\n".join(lines)


# ============== Main Trading Bot ==============

class CryptoTradingBot:
    """Main trading bot with AI decision making - CONSERVATIVE MODE"""
    
    # Top liquid Binance Futures pairs (reduced for quality)
    TRADING_PAIRS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"
    ]
    
    # Conservative strategy parameters
    DEFAULT_LEVERAGE = 3
    DEFAULT_STOP_LOSS_PERCENT = 1.0
    DEFAULT_TAKE_PROFIT_PERCENT = 1.5
    DEFAULT_RISK_PER_TRADE = 3.0  # % of balance per trade
    MIN_SIGNAL_CONFIDENCE = 80  # Only high-quality signals
    
    def __init__(
        self,
        binance_key: str,
        binance_secret: str,
        telegram_token: str,
        telegram_chat_id: str,
        llm_key: str,
        db: AsyncIOMotorClient,
        daily_target_percent: float = 2.5,  # Conservative target
        max_positions: int = 2,  # Fewer positions
        testnet: bool = False
    ):
        self.binance = BinanceFuturesClient(binance_key, binance_secret, testnet)
        self.telegram = TelegramNotifier(telegram_token, telegram_chat_id)
        self.ai_engine = AIStrategyEngine(llm_key)
        self.db = db
        
        self.daily_target_percent = daily_target_percent
        self.max_positions = max_positions
        self.running = False
        
        # Daily tracking
        self.daily_start_balance = 0.0
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.last_report_date = None
    
    async def start(self):
        """Start the trading bot"""
        await self.binance.start()
        self.running = True
        
        # Get initial balance
        self.daily_start_balance = await self.binance.get_balance()
        self.last_report_date = datetime.now(timezone.utc).date()
        
        await self.telegram.send_bot_status(
            "STARTED 🚀",
            f"💰 Balance: ${self.daily_start_balance:,.2f}\n"
            f"🎯 Daily Target: +{self.daily_target_percent}%\n"
            f"📊 Trading {len(self.TRADING_PAIRS)} pairs"
        )
        
        logger.info(f"Bot started with balance: ${self.daily_start_balance:.2f}")
    
    async def stop(self):
        """Stop the trading bot"""
        self.running = False
        await self.binance.stop()
        await self.telegram.send_bot_status("STOPPED 🛑")
        logger.info("Bot stopped")
    
    async def run_trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Check if daily target reached
                current_balance = await self.binance.get_balance()
                daily_pnl_percent = ((current_balance - self.daily_start_balance) / self.daily_start_balance * 100) if self.daily_start_balance > 0 else 0
                
                if daily_pnl_percent >= self.daily_target_percent:
                    await self.telegram.send_message(
                        f"🎉 <b>DAILY TARGET REACHED!</b>\n\n"
                        f"📈 Profit: +{daily_pnl_percent:.2f}%\n"
                        f"💰 Balance: ${current_balance:,.2f}\n\n"
                        f"Bot pausing until next day..."
                    )
                    # Wait until next day
                    await self._wait_for_new_day()
                    continue
                
                # Check for new day
                today = datetime.now(timezone.utc).date()
                if today != self.last_report_date:
                    await self._send_daily_report()
                    self._reset_daily_stats(current_balance)
                    self.last_report_date = today
                
                # Get current positions
                positions = await self.binance.get_open_positions()
                
                # Manage existing positions (check SL/TP)
                await self._manage_positions(positions)
                
                # Look for new opportunities if we have room
                if len(positions) < self.max_positions:
                    await self._find_opportunities(positions, current_balance)
                
                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)
    
    async def _find_opportunities(self, current_positions: List[Position], balance: float):
        """Scan markets for trading opportunities"""
        # Get symbols we're not already in
        current_symbols = {p.symbol for p in current_positions}
        available_pairs = [p for p in self.TRADING_PAIRS if p not in current_symbols]
        
        for symbol in available_pairs:
            if not self.running:
                break
            
            try:
                # Get market data
                price = await self.binance.get_ticker_price(symbol)
                klines = await self.binance.get_klines(symbol, "15m", 100)
                
                # Get AI analysis
                signal = await self.ai_engine.analyze_market(
                    symbol=symbol,
                    klines=klines,
                    current_price=price,
                    balance=balance,
                    positions=current_positions
                )
                
                if signal and signal.confidence >= 70:
                    await self._execute_trade(signal, balance)
                    return  # One trade at a time
                
                await asyncio.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
    
    async def _execute_trade(self, signal: TradeSignal, balance: float):
        """Execute a trade based on AI signal"""
        try:
            # Set leverage
            await self.binance.set_leverage(signal.symbol, signal.leverage)
            
            # Calculate position size
            position_value = balance * (signal.position_size_percent / 100) * signal.leverage
            quantity = position_value / signal.entry_price
            
            # Get precision
            info = await self.binance.get_exchange_info(signal.symbol)
            quantity_precision = 3
            for f in info.get('filters', []):
                if f['filterType'] == 'LOT_SIZE':
                    step = float(f['stepSize'])
                    quantity_precision = len(str(step).rstrip('0').split('.')[-1]) if '.' in str(step) else 0
            
            quantity = float(Decimal(str(quantity)).quantize(Decimal(10) ** -quantity_precision, rounding=ROUND_DOWN))
            
            if quantity <= 0:
                logger.warning(f"Calculated quantity too small for {signal.symbol}")
                return
            
            # Place market order
            order = await self.binance.place_order(
                symbol=signal.symbol,
                side=signal.side.value,
                quantity=quantity
            )
            
            logger.info(f"Order placed: {order}")
            
            # Send Telegram notification
            await self.telegram.send_trade_notification(signal, "NEW TRADE OPENED")
            
            # Record trade
            await self._record_trade(signal, quantity, "OPEN")
            self.trades_today += 1
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            await self.telegram.send_message(f"⚠️ Trade execution failed: {str(e)[:100]}")
    
    async def _manage_positions(self, positions: List[Position]):
        """Monitor and manage open positions"""
        for position in positions:
            try:
                # Check unrealized PnL percentage
                entry_value = position.entry_price * position.quantity
                pnl_percent = (position.unrealized_pnl / entry_value * 100) if entry_value > 0 else 0
                
                # Take profit at +1.5% or stop loss at -2%
                should_close = False
                close_reason = ""
                
                if pnl_percent >= 1.5:
                    should_close = True
                    close_reason = f"Take Profit (+{pnl_percent:.2f}%)"
                    self.winning_trades += 1
                elif pnl_percent <= -2:
                    should_close = True
                    close_reason = f"Stop Loss ({pnl_percent:.2f}%)"
                    self.losing_trades += 1
                
                if should_close:
                    await self.binance.close_position(
                        symbol=position.symbol,
                        side=position.side,
                        quantity=position.quantity
                    )
                    
                    pnl_emoji = "✅" if position.unrealized_pnl >= 0 else "❌"
                    await self.telegram.send_message(
                        f"{pnl_emoji} <b>POSITION CLOSED</b>\n\n"
                        f"📊 {position.symbol} {position.side}\n"
                        f"💰 PnL: ${position.unrealized_pnl:,.2f} ({pnl_percent:+.2f}%)\n"
                        f"📝 Reason: {close_reason}"
                    )
                    
                    await self._record_trade_close(position)
                    
            except Exception as e:
                logger.error(f"Position management error for {position.symbol}: {e}")
    
    async def _record_trade(self, signal: TradeSignal, quantity: float, action: str):
        """Record trade to database"""
        await self.db.trades.insert_one({
            "symbol": signal.symbol,
            "side": signal.side.value,
            "quantity": quantity,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "leverage": signal.leverage,
            "confidence": signal.confidence,
            "reason": signal.reason,
            "action": action,
            "timestamp": datetime.now(timezone.utc)
        })
    
    async def _record_trade_close(self, position: Position):
        """Record trade close to database"""
        await self.db.trade_closes.insert_one({
            "symbol": position.symbol,
            "side": position.side,
            "quantity": position.quantity,
            "pnl": position.unrealized_pnl,
            "timestamp": datetime.now(timezone.utc)
        })
    
    async def _send_daily_report(self):
        """Send end-of-day report"""
        balance = await self.binance.get_balance()
        pnl = balance - self.daily_start_balance
        pnl_percent = (pnl / self.daily_start_balance * 100) if self.daily_start_balance > 0 else 0
        
        stats = DailyStats(
            date=self.last_report_date.strftime("%Y-%m-%d"),
            starting_balance=self.daily_start_balance,
            current_balance=balance,
            profit_loss=pnl,
            profit_loss_percent=pnl_percent,
            trades_count=self.trades_today,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            target_reached=pnl_percent >= self.daily_target_percent
        )
        
        await self.telegram.send_daily_report(stats)
        
        # Save to database
        await self.db.daily_reports.insert_one(stats.model_dump())
    
    def _reset_daily_stats(self, current_balance: float):
        """Reset daily statistics"""
        self.daily_start_balance = current_balance
        self.trades_today = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    async def _wait_for_new_day(self):
        """Wait until next UTC day"""
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_until_tomorrow = (tomorrow - now).total_seconds()
        
        logger.info(f"Waiting {seconds_until_tomorrow/3600:.1f} hours until next day")
        await asyncio.sleep(seconds_until_tomorrow)


# ============== Bot Manager (for API) ==============

class TradingBotManager:
    """Manager for the trading bot"""
    
    _instance: Optional[CryptoTradingBot] = None
    _task: Optional[asyncio.Task] = None
    
    @classmethod
    async def start_bot(
        cls,
        binance_key: str,
        binance_secret: str,
        telegram_token: str,
        telegram_chat_id: str,
        llm_key: str,
        db: AsyncIOMotorClient
    ) -> bool:
        """Start the trading bot"""
        if cls._instance and cls._instance.running:
            return False
        
        cls._instance = CryptoTradingBot(
            binance_key=binance_key,
            binance_secret=binance_secret,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
            llm_key=llm_key,
            db=db,
            daily_target_percent=10.0,
            max_positions=5,
            testnet=False
        )
        
        await cls._instance.start()
        cls._task = asyncio.create_task(cls._instance.run_trading_loop())
        return True
    
    @classmethod
    async def stop_bot(cls) -> bool:
        """Stop the trading bot"""
        if not cls._instance:
            return False
        
        await cls._instance.stop()
        
        if cls._task:
            cls._task.cancel()
            try:
                await cls._task
            except asyncio.CancelledError:
                pass
        
        cls._instance = None
        cls._task = None
        return True
    
    @classmethod
    def get_status(cls) -> Dict:
        """Get bot status"""
        if not cls._instance:
            return {"running": False}
        
        return {
            "running": cls._instance.running,
            "daily_start_balance": cls._instance.daily_start_balance,
            "trades_today": cls._instance.trades_today,
            "winning_trades": cls._instance.winning_trades,
            "losing_trades": cls._instance.losing_trades
        }
