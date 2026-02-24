"""
Backtesting Engine for AI Crypto Trading Bot
Simulates trading strategy on historical data
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
import random

@dataclass
class BacktestTrade:
    """Represents a single trade in backtest"""
    symbol: str
    side: str  # LONG or SHORT
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0.0
    leverage: int = 10
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""
    confidence: float = 0.0

@dataclass
class BacktestResult:
    """Results of backtest"""
    date: str
    starting_balance: float
    ending_balance: float
    total_pnl: float
    total_pnl_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: float
    best_trade: float
    worst_trade: float
    avg_trade_pnl: float
    trades: List[BacktestTrade] = field(default_factory=list)
    hourly_balance: Dict[int, float] = field(default_factory=dict)
    target_reached: bool = False
    target_reached_time: Optional[str] = None

class BacktestEngine:
    """Backtesting engine for crypto trading strategy"""
    
    TRADING_PAIRS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"
    ]
    
    def __init__(
        self,
        starting_balance: float = 1000.0,
        daily_target_percent: float = 10.0,
        max_positions: int = 5,
        leverage: int = 10,
        stop_loss_percent: float = 2.0,
        take_profit_percent: float = 1.5
    ):
        self.starting_balance = starting_balance
        self.daily_target_percent = daily_target_percent
        self.max_positions = max_positions
        self.leverage = leverage
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        
        self.balance = starting_balance
        self.trades: List[BacktestTrade] = []
        self.open_positions: List[BacktestTrade] = []
        self.hourly_balance: Dict[int, float] = {}
        self.historical_data: Dict[str, List] = {}
    
    async def fetch_historical_data(self, symbol: str, date: datetime) -> List:
        """Fetch 15m klines for a specific date from Binance"""
        start_time = int(datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        end_time = int(datetime(date.year, date.month, date.day, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000)
        
        url = f"https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": "15m",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 96  # 24 hours * 4 (15min intervals)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
    
    def calculate_indicators(self, klines: List, index: int) -> Dict:
        """Calculate technical indicators at a given point"""
        if index < 20:
            return {}
        
        closes = [float(k[4]) for k in klines[:index+1]]
        highs = [float(k[2]) for k in klines[:index+1]]
        lows = [float(k[3]) for k in klines[:index+1]]
        volumes = [float(k[5]) for k in klines[:index+1]]
        
        # SMAs
        sma_10 = sum(closes[-10:]) / 10
        sma_20 = sum(closes[-20:]) / 20
        
        # RSI
        gains, losses = [], []
        for i in range(1, min(15, len(closes))):
            diff = closes[-i] - closes[-i-1]
            if diff > 0:
                gains.append(diff)
            else:
                losses.append(abs(diff))
        
        avg_gain = sum(gains) / 14 if gains else 0
        avg_loss = sum(losses) / 14 if losses else 0.0001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Volume ratio
        avg_vol = sum(volumes[-20:]) / 20
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1
        
        # Price change
        price_change = (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) >= 4 else 0
        
        return {
            "sma_10": sma_10,
            "sma_20": sma_20,
            "rsi": rsi,
            "volume_ratio": vol_ratio,
            "price_change_1h": price_change,
            "current_price": closes[-1],
            "trend": "BULLISH" if sma_10 > sma_20 else "BEARISH"
        }
    
    def generate_signal(self, indicators: Dict, symbol: str) -> Optional[Tuple[str, float]]:
        """Generate trading signal based on indicators (simulating AI decision)"""
        if not indicators:
            return None
        
        rsi = indicators["rsi"]
        trend = indicators["trend"]
        vol_ratio = indicators["volume_ratio"]
        price_change = indicators["price_change_1h"]
        
        confidence = 0
        side = None
        
        # Bullish signals
        if trend == "BULLISH" and rsi < 40 and vol_ratio > 1.2:
            side = "LONG"
            confidence = 70 + min(rsi, 20) + min(vol_ratio * 5, 15)
        
        # Oversold bounce
        elif rsi < 30 and price_change < -1:
            side = "LONG"
            confidence = 65 + (30 - rsi) + abs(price_change)
        
        # Bearish signals
        elif trend == "BEARISH" and rsi > 60 and vol_ratio > 1.2:
            side = "SHORT"
            confidence = 70 + min(100 - rsi, 20) + min(vol_ratio * 5, 15)
        
        # Overbought reversal
        elif rsi > 70 and price_change > 1:
            side = "SHORT"
            confidence = 65 + (rsi - 70) + price_change
        
        # Add some randomness to simulate AI variance
        if side and confidence >= 70:
            confidence = min(confidence + random.uniform(-5, 10), 95)
            return (side, confidence)
        
        return None
    
    def check_exit_conditions(self, trade: BacktestTrade, current_price: float) -> Optional[Tuple[str, float]]:
        """Check if position should be closed"""
        if trade.side == "LONG":
            pnl_percent = (current_price - trade.entry_price) / trade.entry_price * 100 * trade.leverage
            
            if pnl_percent >= self.take_profit_percent * trade.leverage:
                return ("TAKE_PROFIT", pnl_percent)
            elif pnl_percent <= -self.stop_loss_percent * trade.leverage:
                return ("STOP_LOSS", pnl_percent)
        else:  # SHORT
            pnl_percent = (trade.entry_price - current_price) / trade.entry_price * 100 * trade.leverage
            
            if pnl_percent >= self.take_profit_percent * trade.leverage:
                return ("TAKE_PROFIT", pnl_percent)
            elif pnl_percent <= -self.stop_loss_percent * trade.leverage:
                return ("STOP_LOSS", pnl_percent)
        
        return None
    
    async def run_backtest(self, date: datetime) -> BacktestResult:
        """Run backtest for a specific date"""
        print(f"\n🔄 Running backtest for {date.strftime('%Y-%m-%d')}...")
        print(f"💰 Starting balance: ${self.starting_balance:,.2f}")
        print(f"🎯 Daily target: +{self.daily_target_percent}%")
        print(f"⚡ Leverage: {self.leverage}x | SL: {self.stop_loss_percent}% | TP: {self.take_profit_percent}%")
        print("-" * 60)
        
        # Fetch historical data for all pairs
        print("📊 Fetching historical data...")
        for symbol in self.TRADING_PAIRS:
            data = await self.fetch_historical_data(symbol, date)
            if data:
                self.historical_data[symbol] = data
                print(f"  ✓ {symbol}: {len(data)} candles")
        
        if not self.historical_data:
            print("❌ No historical data available")
            return BacktestResult(
                date=date.strftime('%Y-%m-%d'),
                starting_balance=self.starting_balance,
                ending_balance=self.starting_balance,
                total_pnl=0,
                total_pnl_percent=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                max_drawdown=0,
                best_trade=0,
                worst_trade=0,
                avg_trade_pnl=0
            )
        
        # Reset state
        self.balance = self.starting_balance
        self.trades = []
        self.open_positions = []
        self.hourly_balance = {0: self.starting_balance}
        
        target_reached = False
        target_reached_time = None
        max_balance = self.starting_balance
        min_balance = self.starting_balance
        
        # Get number of candles (should be 96 for 24h with 15m intervals)
        num_candles = max(len(data) for data in self.historical_data.values())
        
        print(f"\n⏱️ Simulating {num_candles} time periods...\n")
        
        # Iterate through each 15-minute candle
        for i in range(20, num_candles):  # Start at 20 to have enough data for indicators
            current_time = datetime.fromtimestamp(
                int(list(self.historical_data.values())[0][i][0]) / 1000, 
                tz=timezone.utc
            )
            
            # Check if daily target reached
            current_pnl_percent = (self.balance - self.starting_balance) / self.starting_balance * 100
            if current_pnl_percent >= self.daily_target_percent and not target_reached:
                target_reached = True
                target_reached_time = current_time.strftime('%H:%M')
                print(f"🎉 TARGET REACHED at {target_reached_time}! P&L: +{current_pnl_percent:.2f}%")
                # Bot would pause here in real trading, but we continue to see full day performance
            
            # Track hourly balance
            hour = current_time.hour
            self.hourly_balance[hour] = self.balance
            
            # Track max/min balance for drawdown
            max_balance = max(max_balance, self.balance)
            min_balance = min(min_balance, self.balance)
            
            # Check existing positions
            positions_to_close = []
            for trade in self.open_positions:
                if trade.symbol in self.historical_data:
                    current_price = float(self.historical_data[trade.symbol][i][4])  # Close price
                    exit_check = self.check_exit_conditions(trade, current_price)
                    
                    if exit_check:
                        exit_reason, pnl_percent = exit_check
                        trade.exit_time = current_time
                        trade.exit_price = current_price
                        trade.exit_reason = exit_reason
                        
                        # Calculate actual PnL
                        position_size = self.starting_balance * 0.05  # 5% per trade
                        trade.pnl = position_size * (pnl_percent / 100)
                        trade.pnl_percent = pnl_percent
                        
                        self.balance += trade.pnl
                        positions_to_close.append(trade)
                        
                        emoji = "✅" if trade.pnl > 0 else "❌"
                        print(f"{emoji} [{current_time.strftime('%H:%M')}] CLOSE {trade.symbol} {trade.side}: "
                              f"${trade.pnl:+.2f} ({pnl_percent:+.2f}%) - {exit_reason}")
            
            # Remove closed positions
            for trade in positions_to_close:
                self.open_positions.remove(trade)
                self.trades.append(trade)
            
            # Look for new opportunities if we have room
            if len(self.open_positions) < self.max_positions:
                # Check each symbol for signals
                for symbol in self.TRADING_PAIRS:
                    if symbol not in self.historical_data:
                        continue
                    
                    # Skip if already in this position
                    if any(t.symbol == symbol for t in self.open_positions):
                        continue
                    
                    klines = self.historical_data[symbol][:i+1]
                    indicators = self.calculate_indicators(klines, i)
                    signal = self.generate_signal(indicators, symbol)
                    
                    if signal:
                        side, confidence = signal
                        current_price = float(klines[i][4])
                        
                        # Open new position
                        trade = BacktestTrade(
                            symbol=symbol,
                            side=side,
                            entry_time=current_time,
                            entry_price=current_price,
                            leverage=self.leverage,
                            confidence=confidence,
                            quantity=self.starting_balance * 0.05 / current_price
                        )
                        
                        self.open_positions.append(trade)
                        print(f"🔵 [{current_time.strftime('%H:%M')}] OPEN {symbol} {side} @ ${current_price:,.2f} "
                              f"(confidence: {confidence:.0f}%)")
                        
                        # Only open one position per candle
                        break
        
        # Close any remaining positions at end of day
        if self.open_positions:
            print(f"\n📍 Closing {len(self.open_positions)} remaining positions at end of day...")
            for trade in self.open_positions:
                if trade.symbol in self.historical_data:
                    final_price = float(self.historical_data[trade.symbol][-1][4])
                    trade.exit_time = current_time
                    trade.exit_price = final_price
                    trade.exit_reason = "END_OF_DAY"
                    
                    if trade.side == "LONG":
                        pnl_percent = (final_price - trade.entry_price) / trade.entry_price * 100 * trade.leverage
                    else:
                        pnl_percent = (trade.entry_price - final_price) / trade.entry_price * 100 * trade.leverage
                    
                    position_size = self.starting_balance * 0.05
                    trade.pnl = position_size * (pnl_percent / 100)
                    trade.pnl_percent = pnl_percent
                    
                    self.balance += trade.pnl
                    self.trades.append(trade)
                    
                    emoji = "✅" if trade.pnl > 0 else "❌"
                    print(f"{emoji} CLOSE {trade.symbol}: ${trade.pnl:+.2f} ({pnl_percent:+.2f}%)")
        
        # Calculate results
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = self.balance - self.starting_balance
        total_pnl_percent = (total_pnl / self.starting_balance) * 100
        
        max_drawdown = ((max_balance - min_balance) / max_balance * 100) if max_balance > 0 else 0
        
        result = BacktestResult(
            date=date.strftime('%Y-%m-%d'),
            starting_balance=self.starting_balance,
            ending_balance=self.balance,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=(len(winning_trades) / len(self.trades) * 100) if self.trades else 0,
            max_drawdown=max_drawdown,
            best_trade=max(t.pnl for t in self.trades) if self.trades else 0,
            worst_trade=min(t.pnl for t in self.trades) if self.trades else 0,
            avg_trade_pnl=(sum(t.pnl for t in self.trades) / len(self.trades)) if self.trades else 0,
            trades=self.trades,
            hourly_balance=self.hourly_balance,
            target_reached=target_reached,
            target_reached_time=target_reached_time
        )
        
        return result
    
    def print_results(self, result: BacktestResult):
        """Print backtest results"""
        print("\n" + "=" * 60)
        print(f"📊 BACKTEST RESULTS - {result.date}")
        print("=" * 60)
        
        # Target status
        if result.target_reached:
            print(f"\n🎉 DAILY TARGET REACHED at {result.target_reached_time}!")
        else:
            print(f"\n⚠️ Daily target (+{self.daily_target_percent}%) NOT reached")
        
        # P&L Summary
        pnl_emoji = "📈" if result.total_pnl >= 0 else "📉"
        print(f"\n{pnl_emoji} PROFIT & LOSS:")
        print(f"   Starting Balance: ${result.starting_balance:,.2f}")
        print(f"   Ending Balance:   ${result.ending_balance:,.2f}")
        print(f"   Total P&L:        ${result.total_pnl:+,.2f} ({result.total_pnl_percent:+.2f}%)")
        
        # Trade Statistics
        print(f"\n📊 TRADE STATISTICS:")
        print(f"   Total Trades:     {result.total_trades}")
        print(f"   Winning Trades:   {result.winning_trades} ✅")
        print(f"   Losing Trades:    {result.losing_trades} ❌")
        print(f"   Win Rate:         {result.win_rate:.1f}%")
        
        # Risk Metrics
        print(f"\n⚠️ RISK METRICS:")
        print(f"   Max Drawdown:     {result.max_drawdown:.2f}%")
        print(f"   Best Trade:       ${result.best_trade:+.2f}")
        print(f"   Worst Trade:      ${result.worst_trade:+.2f}")
        print(f"   Avg Trade P&L:    ${result.avg_trade_pnl:+.2f}")
        
        # Hourly Balance Chart (ASCII)
        print(f"\n📈 HOURLY BALANCE:")
        if result.hourly_balance:
            min_bal = min(result.hourly_balance.values())
            max_bal = max(result.hourly_balance.values())
            range_bal = max_bal - min_bal if max_bal != min_bal else 1
            
            for hour in sorted(result.hourly_balance.keys()):
                bal = result.hourly_balance[hour]
                bar_length = int((bal - min_bal) / range_bal * 30) + 1
                bar = "█" * bar_length
                change = (bal - result.starting_balance) / result.starting_balance * 100
                print(f"   {hour:02d}:00 | {bar} ${bal:,.0f} ({change:+.1f}%)")
        
        # Individual Trades
        print(f"\n📝 TRADE LOG:")
        print("-" * 80)
        print(f"{'Time':<8} {'Symbol':<10} {'Side':<6} {'Entry':<12} {'Exit':<12} {'P&L':<12} {'Reason':<12}")
        print("-" * 80)
        
        for trade in result.trades:
            entry_time = trade.entry_time.strftime('%H:%M') if trade.entry_time else "N/A"
            pnl_str = f"${trade.pnl:+.2f}" if trade.pnl else "N/A"
            print(f"{entry_time:<8} {trade.symbol:<10} {trade.side:<6} "
                  f"${trade.entry_price:>10,.2f} ${trade.exit_price:>10,.2f} "
                  f"{pnl_str:<12} {trade.exit_reason:<12}")
        
        print("=" * 60)


async def main():
    """Run backtest for February 23, 2025"""
    engine = BacktestEngine(
        starting_balance=1000.0,
        daily_target_percent=10.0,
        max_positions=5,
        leverage=10,
        stop_loss_percent=2.0,
        take_profit_percent=1.5
    )
    
    # February 23, 2025
    target_date = datetime(2025, 2, 23, tzinfo=timezone.utc)
    
    result = await engine.run_backtest(target_date)
    engine.print_results(result)
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
