"""
Backtest with synthetic data based on real market patterns
Simulates February 23, 2025 trading day
"""

import asyncio
import random
import math
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Seed for reproducibility
random.seed(20250223)

@dataclass
class BacktestTrade:
    """Represents a single trade in backtest"""
    symbol: str
    side: str
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


def generate_price_data(base_price: float, volatility: float, num_candles: int, trend: float = 0) -> List[Dict]:
    """Generate realistic OHLCV data with patterns"""
    candles = []
    price = base_price
    
    for i in range(num_candles):
        # Add trend component
        trend_effect = trend * (i / num_candles)
        
        # Add some intraday patterns (higher volatility during certain hours)
        hour = (i * 15 // 60) % 24
        if 8 <= hour <= 12 or 14 <= hour <= 18:  # High activity hours
            period_vol = volatility * 1.5
        else:
            period_vol = volatility * 0.7
        
        # Generate OHLCV
        change = random.gauss(trend_effect / num_candles, period_vol)
        open_price = price
        
        # Random walk for high/low
        high_price = open_price * (1 + abs(random.gauss(0, period_vol * 0.5)))
        low_price = open_price * (1 - abs(random.gauss(0, period_vol * 0.5)))
        
        # Close somewhere between high and low
        close_price = low_price + random.random() * (high_price - low_price)
        close_price = open_price * (1 + change)
        
        # Ensure OHLC consistency
        high_price = max(open_price, close_price, high_price)
        low_price = min(open_price, close_price, low_price)
        
        # Volume with patterns
        base_volume = base_price * 1000
        volume = base_volume * (0.5 + random.random()) * (1.5 if 8 <= hour <= 18 else 0.8)
        
        candles.append({
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "timestamp": datetime(2025, 2, 23, hour, (i * 15) % 60, tzinfo=timezone.utc)
        })
        
        price = close_price
    
    return candles


class BacktestEngine:
    """Backtesting engine with synthetic data"""
    
    # Market data for Feb 23, 2025 (approximate based on market conditions)
    MARKET_DATA = {
        "BTCUSDT": {"base": 96500, "volatility": 0.008, "trend": 0.02},   # Slight bullish
        "ETHUSDT": {"base": 2780, "volatility": 0.012, "trend": 0.015},   # Moderate bullish
        "BNBUSDT": {"base": 650, "volatility": 0.010, "trend": -0.01},    # Slight bearish
        "SOLUSDT": {"base": 168, "volatility": 0.015, "trend": 0.03},     # Strong bullish
        "XRPUSDT": {"base": 2.65, "volatility": 0.014, "trend": 0.01},    # Slight bullish
        "DOGEUSDT": {"base": 0.25, "volatility": 0.018, "trend": -0.02},  # Bearish
        "ADAUSDT": {"base": 0.78, "volatility": 0.013, "trend": 0.005},   # Neutral
        "AVAXUSDT": {"base": 38.5, "volatility": 0.016, "trend": 0.025},  # Bullish
        "LINKUSDT": {"base": 18.2, "volatility": 0.014, "trend": 0.02},   # Bullish
        "DOTUSDT": {"base": 7.8, "volatility": 0.012, "trend": -0.015},   # Bearish
    }
    
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
        self.price_data: Dict[str, List[Dict]] = {}
    
    def generate_all_price_data(self):
        """Generate price data for all symbols"""
        num_candles = 96  # 24 hours * 4 (15min candles)
        
        for symbol, params in self.MARKET_DATA.items():
            self.price_data[symbol] = generate_price_data(
                base_price=params["base"],
                volatility=params["volatility"],
                num_candles=num_candles,
                trend=params["trend"]
            )
    
    def calculate_indicators(self, candles: List[Dict], index: int) -> Dict:
        """Calculate technical indicators"""
        if index < 20:
            return {}
        
        closes = [c["close"] for c in candles[:index+1]]
        volumes = [c["volume"] for c in candles[:index+1]]
        
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
        """AI-like signal generation"""
        if not indicators:
            return None
        
        rsi = indicators["rsi"]
        trend = indicators["trend"]
        vol_ratio = indicators["volume_ratio"]
        price_change = indicators["price_change_1h"]
        
        confidence = 0
        side = None
        
        # Strong bullish signals
        if trend == "BULLISH" and rsi < 45 and vol_ratio > 1.1:
            side = "LONG"
            confidence = 72 + min(45 - rsi, 15) + min(vol_ratio * 3, 10)
        
        # Oversold bounce
        elif rsi < 32 and price_change < -0.8:
            side = "LONG"
            confidence = 70 + (32 - rsi) * 0.8 + abs(price_change) * 2
        
        # Strong bearish signals
        elif trend == "BEARISH" and rsi > 55 and vol_ratio > 1.1:
            side = "SHORT"
            confidence = 72 + min(rsi - 55, 15) + min(vol_ratio * 3, 10)
        
        # Overbought reversal
        elif rsi > 68 and price_change > 0.8:
            side = "SHORT"
            confidence = 70 + (rsi - 68) * 0.8 + price_change * 2
        
        # Add variance
        if side and confidence >= 70:
            confidence = min(confidence + random.uniform(-3, 5), 92)
            return (side, confidence)
        
        return None
    
    def check_exit(self, trade: BacktestTrade, current_price: float) -> Optional[Tuple[str, float]]:
        """Check exit conditions"""
        if trade.side == "LONG":
            pnl_percent = (current_price - trade.entry_price) / trade.entry_price * 100 * trade.leverage
        else:
            pnl_percent = (trade.entry_price - current_price) / trade.entry_price * 100 * trade.leverage
        
        if pnl_percent >= self.take_profit_percent * trade.leverage:
            return ("TAKE_PROFIT", pnl_percent)
        elif pnl_percent <= -self.stop_loss_percent * trade.leverage:
            return ("STOP_LOSS", pnl_percent)
        
        return None
    
    def run_backtest(self) -> BacktestResult:
        """Run the backtest"""
        print("\n" + "🤖 AI CRYPTO SCALPER - BACKTEST".center(60, "═"))
        print(f"\n📅 Date: February 23, 2025")
        print(f"💰 Starting Balance: ${self.starting_balance:,.2f}")
        print(f"🎯 Daily Target: +{self.daily_target_percent}%")
        print(f"⚡ Leverage: {self.leverage}x")
        print(f"🛡️ Risk: SL {self.stop_loss_percent}% | TP {self.take_profit_percent}%")
        print(f"📊 Max Positions: {self.max_positions}")
        print("─" * 60)
        
        # Generate synthetic data
        print("\n📊 Generating market data...")
        self.generate_all_price_data()
        
        for symbol in self.price_data:
            data = self.price_data[symbol]
            start_price = data[0]["close"]
            end_price = data[-1]["close"]
            change = (end_price - start_price) / start_price * 100
            emoji = "📈" if change > 0 else "📉"
            print(f"  {emoji} {symbol}: ${start_price:,.2f} → ${end_price:,.2f} ({change:+.2f}%)")
        
        # Reset state
        self.balance = self.starting_balance
        self.trades = []
        self.open_positions = []
        self.hourly_balance = {0: self.starting_balance}
        
        target_reached = False
        target_reached_time = None
        max_balance = self.starting_balance
        
        num_candles = 96
        
        print("\n⏱️ Running simulation...\n")
        print("─" * 60)
        
        for i in range(20, num_candles):
            # Get current time
            hour = (i * 15) // 60
            minute = (i * 15) % 60
            current_time = datetime(2025, 2, 23, hour, minute, tzinfo=timezone.utc)
            
            # Check target
            pnl_percent = (self.balance - self.starting_balance) / self.starting_balance * 100
            if pnl_percent >= self.daily_target_percent and not target_reached:
                target_reached = True
                target_reached_time = f"{hour:02d}:{minute:02d}"
                print(f"\n🎉 TARGET REACHED at {target_reached_time}! Balance: ${self.balance:,.2f} (+{pnl_percent:.2f}%)\n")
            
            # Track hourly
            self.hourly_balance[hour] = self.balance
            max_balance = max(max_balance, self.balance)
            
            # Check open positions
            closed = []
            for trade in self.open_positions:
                current_price = self.price_data[trade.symbol][i]["close"]
                exit_check = self.check_exit(trade, current_price)
                
                if exit_check:
                    reason, pnl_pct = exit_check
                    trade.exit_time = current_time
                    trade.exit_price = current_price
                    trade.exit_reason = reason
                    
                    position_size = self.starting_balance * 0.05
                    trade.pnl = position_size * (pnl_pct / 100)
                    trade.pnl_percent = pnl_pct
                    
                    self.balance += trade.pnl
                    closed.append(trade)
                    
                    emoji = "✅" if trade.pnl > 0 else "❌"
                    print(f"{emoji} [{current_time.strftime('%H:%M')}] CLOSE {trade.symbol} {trade.side}: "
                          f"${trade.pnl:+.2f} ({pnl_pct:+.1f}%) - {reason}")
            
            for t in closed:
                self.open_positions.remove(t)
                self.trades.append(t)
            
            # Look for new trades
            if len(self.open_positions) < self.max_positions and not target_reached:
                for symbol in self.price_data:
                    if any(t.symbol == symbol for t in self.open_positions):
                        continue
                    
                    candles = self.price_data[symbol][:i+1]
                    indicators = self.calculate_indicators(candles, i)
                    signal = self.generate_signal(indicators, symbol)
                    
                    if signal:
                        side, confidence = signal
                        current_price = candles[i]["close"]
                        
                        trade = BacktestTrade(
                            symbol=symbol,
                            side=side,
                            entry_time=current_time,
                            entry_price=current_price,
                            leverage=self.leverage,
                            confidence=confidence
                        )
                        
                        self.open_positions.append(trade)
                        print(f"🔵 [{current_time.strftime('%H:%M')}] OPEN {symbol} {side} @ ${current_price:,.2f} "
                              f"(conf: {confidence:.0f}%)")
                        break
        
        # Close remaining positions
        if self.open_positions:
            print(f"\n📍 Closing {len(self.open_positions)} positions at EOD...")
            for trade in self.open_positions:
                final_price = self.price_data[trade.symbol][-1]["close"]
                trade.exit_time = current_time
                trade.exit_price = final_price
                trade.exit_reason = "END_OF_DAY"
                
                if trade.side == "LONG":
                    pnl_pct = (final_price - trade.entry_price) / trade.entry_price * 100 * trade.leverage
                else:
                    pnl_pct = (trade.entry_price - final_price) / trade.entry_price * 100 * trade.leverage
                
                position_size = self.starting_balance * 0.05
                trade.pnl = position_size * (pnl_pct / 100)
                trade.pnl_percent = pnl_pct
                self.balance += trade.pnl
                self.trades.append(trade)
                
                emoji = "✅" if trade.pnl > 0 else "❌"
                print(f"{emoji} CLOSE {trade.symbol}: ${trade.pnl:+.2f} ({pnl_pct:+.1f}%)")
        
        # Calculate results
        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = self.balance - self.starting_balance
        total_pnl_pct = total_pnl / self.starting_balance * 100
        
        min_balance = min(self.hourly_balance.values())
        max_dd = (max_balance - min_balance) / max_balance * 100
        
        result = BacktestResult(
            date="2025-02-23",
            starting_balance=self.starting_balance,
            ending_balance=self.balance,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_pct,
            total_trades=len(self.trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / len(self.trades) * 100 if self.trades else 0,
            max_drawdown=max_dd,
            best_trade=max(t.pnl for t in self.trades) if self.trades else 0,
            worst_trade=min(t.pnl for t in self.trades) if self.trades else 0,
            avg_trade_pnl=sum(t.pnl for t in self.trades) / len(self.trades) if self.trades else 0,
            trades=self.trades,
            hourly_balance=self.hourly_balance,
            target_reached=target_reached,
            target_reached_time=target_reached_time
        )
        
        return result
    
    def print_results(self, result: BacktestResult):
        """Print final results"""
        print("\n" + "═" * 60)
        print("📊 BACKTEST RESULTS - February 23, 2025".center(60))
        print("═" * 60)
        
        if result.target_reached:
            print(f"\n🎉 DAILY TARGET (+{self.daily_target_percent}%) REACHED at {result.target_reached_time}!")
        else:
            print(f"\n⚠️ Daily target (+{self.daily_target_percent}%) not reached")
        
        pnl_emoji = "📈" if result.total_pnl >= 0 else "📉"
        print(f"\n{pnl_emoji} PROFIT & LOSS:")
        print(f"   💵 Starting:  ${result.starting_balance:,.2f}")
        print(f"   💰 Ending:    ${result.ending_balance:,.2f}")
        print(f"   📊 Total P&L: ${result.total_pnl:+,.2f} ({result.total_pnl_percent:+.2f}%)")
        
        print(f"\n📈 TRADE STATISTICS:")
        print(f"   Total Trades:   {result.total_trades}")
        print(f"   ✅ Winners:     {result.winning_trades}")
        print(f"   ❌ Losers:      {result.losing_trades}")
        print(f"   🎯 Win Rate:    {result.win_rate:.1f}%")
        
        print(f"\n⚠️ RISK METRICS:")
        print(f"   Max Drawdown:   {result.max_drawdown:.2f}%")
        print(f"   Best Trade:     ${result.best_trade:+.2f}")
        print(f"   Worst Trade:    ${result.worst_trade:+.2f}")
        print(f"   Avg Trade:      ${result.avg_trade_pnl:+.2f}")
        
        # Hourly performance chart
        print(f"\n📈 HOURLY BALANCE CURVE:")
        min_bal = min(result.hourly_balance.values())
        max_bal = max(result.hourly_balance.values())
        range_bal = max_bal - min_bal if max_bal != min_bal else 1
        
        for hour in range(24):
            if hour in result.hourly_balance:
                bal = result.hourly_balance[hour]
                bar_len = int((bal - min_bal) / range_bal * 25) + 1
                pct = (bal - self.starting_balance) / self.starting_balance * 100
                bar = "█" * bar_len
                color = "+" if pct >= 0 else ""
                print(f"   {hour:02d}:00 │{bar:<25} ${bal:,.0f} ({color}{pct:.1f}%)")
        
        # Trade summary table
        print(f"\n📝 TRADE LOG:")
        print("─" * 85)
        print(f"{'Time':<6} {'Symbol':<10} {'Side':<6} {'Entry':>10} {'Exit':>10} {'P&L':>10} {'%':>8} {'Reason':<12}")
        print("─" * 85)
        
        for t in result.trades:
            time_str = t.entry_time.strftime('%H:%M')
            pnl_str = f"${t.pnl:+.2f}"
            pct_str = f"{t.pnl_percent:+.1f}%"
            print(f"{time_str:<6} {t.symbol:<10} {t.side:<6} ${t.entry_price:>9,.2f} ${t.exit_price:>9,.2f} {pnl_str:>10} {pct_str:>8} {t.exit_reason:<12}")
        
        print("─" * 85)
        print(f"{'TOTAL':<6} {'':<10} {'':<6} {'':<10} {'':<10} ${result.total_pnl:>+9.2f} {result.total_pnl_percent:>+7.1f}%")
        print("═" * 85)


def main():
    engine = BacktestEngine(
        starting_balance=1000.0,
        daily_target_percent=10.0,
        max_positions=5,
        leverage=10,
        stop_loss_percent=2.0,
        take_profit_percent=1.5
    )
    
    result = engine.run_backtest()
    engine.print_results(result)


if __name__ == "__main__":
    main()
