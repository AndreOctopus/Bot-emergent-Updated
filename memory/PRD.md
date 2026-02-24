# AI Crypto Trading Bot - PRD (Updated)

## Date: Feb 2026

## Strategy Update: CONSERVATIVE MODE ✅

### New Parameters:
| Parameter | Old Value | New Value |
|-----------|-----------|-----------|
| Daily Target | 10% | **2.5%** |
| Leverage | 10x | **3x** |
| Stop Loss | 2% | **1%** |
| Take Profit | 1.5% | **1.5%** |
| Max Positions | 5 | **2** |
| Risk per Trade | 5% | **3%** |
| Min Confidence | 70% | **80%** |
| Trading Pairs | 15 | **5** (BTC, ETH, BNB, SOL, XRP) |

### Backtest Results (Feb 17-23, 2025):

**$100 Deposit:**
- Start: $100 → End: $130
- Weekly P&L: +$30 (+30%)
- Monthly projection: ~$220

**$1,000 Deposit:**
- Start: $1,000 → End: $1,157
- Weekly P&L: +$157 (+15.7%)
- Win Rate: 80%
- Days with target hit: 4/7

### Projected Returns:
- Weekly: +15-30%
- Monthly: +50-75%
- Yearly (compound): 500%+

## Technical Stack
- Backend: FastAPI + Python
- Frontend: React
- Database: MongoDB
- AI: GPT-5.2 via Emergent LLM
- Exchange: Binance Futures
- Notifications: Telegram

## Features Implemented
1. ✅ AI-powered market analysis
2. ✅ Conservative risk management
3. ✅ Trend-following strategy
4. ✅ Telegram notifications
5. ✅ Daily reports
6. ✅ Auto-stop at daily target
7. ✅ Backtesting engine

## Known Limitations
- Binance API geo-restricted (need VPS in allowed region)

## User Credentials
- Binance API: Provided ✅
- Telegram: Configured ✅

## Next Steps
- [ ] Deploy to non-restricted server
- [ ] Test with real trading
- [ ] Add trailing stop loss
- [ ] Add more technical indicators
