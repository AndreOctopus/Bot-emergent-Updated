# AI Crypto Trading Bot - PRD

## Date: Jan 2026

## Original Problem Statement
Create an autonomous AI trading bot for Binance Futures that:
- Analyzes market daily and selects best trades
- Opens trades and continues trading until +10% daily profit
- Uses intraday scalping strategy
- Makes independent decisions with AI
- Has strategy and risk management
- Sends all trades and reports to Telegram

## Architecture
- **Backend**: FastAPI (Python)
- **Frontend**: React
- **Database**: MongoDB
- **AI Engine**: GPT-5.2 via Emergent LLM
- **Exchange**: Binance Futures API
- **Notifications**: Telegram Bot API

## What's Been Implemented

### Core Components
1. **BinanceFuturesClient** - Full async client for Binance Futures
   - Account balance & positions
   - Order placement (market, limit, stop-loss)
   - Real-time price data
   - Leverage management

2. **AIStrategyEngine** - AI-powered decision making
   - Technical analysis (SMA, RSI, ATR, volume)
   - GPT-5.2 analysis for trade signals
   - Confidence scoring (0-100%)
   - Dynamic position sizing

3. **TelegramNotifier** - Real-time notifications
   - Trade entry/exit alerts
   - Daily performance reports
   - Bot status updates

4. **CryptoTradingBot** - Main orchestrator
   - 15 liquid trading pairs
   - Max 5 concurrent positions
   - 10% daily profit target
   - Auto-pause when target reached
   - Risk management (SL: 2%, TP: 1.5%)

5. **Trading Dashboard** (React)
   - Real-time stats display
   - Configuration panel
   - Trade history
   - Daily reports
   - Target progress bar

### API Endpoints
- `POST /api/trading/start` - Start bot with config
- `POST /api/trading/stop` - Stop bot
- `GET /api/trading/status` - Get bot status
- `GET /api/trading/trades` - Get recent trades
- `GET /api/trading/reports` - Get daily reports

## Configuration Required
- Binance API Key & Secret
- Telegram Bot Token
- Telegram Chat ID

## Known Limitations
1. Binance API geo-restricted from some server locations
2. Requires VPN/proxy from restricted regions

## User Credentials Provided
- Binance keys: Provided ✅
- Telegram token: 8144215710:AAEX_U3V2HQhJ5AhVpFJbt5CRm2dA5yDiHg ✅
- Telegram chat: -1003844330472 ✅

## Next Steps (P0)
- [ ] Add VPN/proxy support for Binance API
- [ ] Test with real trading
- [ ] Add position sizing customization

## Future Enhancements (P1/P2)
- WebSocket real-time prices
- Multiple strategy modes
- Backtesting engine
- Mobile app notifications
- Portfolio analytics

## Risk Warning
⚠️ Cryptocurrency futures trading involves substantial risk. 
Use leverage responsibly. Never trade more than you can afford to lose.
