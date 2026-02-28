"""
AI Crypto Trading Bot - FastAPI Server
All API keys hardcoded for easy Railway deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from datetime import datetime, timezone
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== HARDCODED CONFIGURATION ==============
# MongoDB - Railway will provide this, fallback to local
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
DB_NAME = "trading_bot"

# OpenAI API Key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', "fallback_key_here")

# Binance Futures API
BINANCE_API_KEY = "cuCbg2Yz9Co9lBCSUFHGuGqfK44s69NQOKRsbgJD97KzUIv6KsQCC0u6t4zc1a3I"
BINANCE_SECRET_KEY = "Vz8VPUugXwaAfIFh6NmIb6OGu8cOIh1OZuoEgsebIl08UPyAn6ErUYAjfLmV01Xa"

# Telegram Bot
TELEGRAM_BOT_TOKEN = "8144215710:AAEX_U3V2HQhJ5AhVpFJbt5CRm2dA5yDiHg"
TELEGRAM_CHAT_ID = "-1003844330472"
# =====================================================

# MongoDB connection
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# FastAPI app
app = FastAPI(title="AI Crypto Trading Bot", version="1.0.0")

# CORS - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Models ==============

class TradingBotStartRequest(BaseModel):
    # Optional - will use hardcoded values if not provided
    binance_api_key: str = None
    binance_secret_key: str = None
    telegram_token: str = None
    telegram_chat_id: str = None


# ============== Bot State ==============

class BotState:
    running = False
    daily_start_balance = 0.0
    trades_today = 0
    winning_trades = 0
    losing_trades = 0

bot_state = BotState()


# ============== Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint"""
    return {"name": "AI Crypto Trading Bot", "status": "running", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    """Health check for Railway"""
    try:
        await db.command('ping')
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/status")
async def get_status():
    """Get API status"""
    return {
        "status": "ok",
        "bot_running": bot_state.running,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/api/trading/start")
async def start_trading_bot(request: TradingBotStartRequest = None):
    """Start the trading bot with hardcoded or provided credentials"""
    
    if bot_state.running:
        return {"ok": False, "message": "Bot is already running"}
    
    # Use hardcoded values or provided ones
    binance_key = request.binance_api_key if request and request.binance_api_key else BINANCE_API_KEY
    binance_secret = request.binance_secret_key if request and request.binance_secret_key else BINANCE_SECRET_KEY
    telegram_token = request.telegram_token if request and request.telegram_token else TELEGRAM_BOT_TOKEN
    telegram_chat = request.telegram_chat_id if request and request.telegram_chat_id else TELEGRAM_CHAT_ID
    
    try:
        from trading_bot import TradingBotManager
        success = await TradingBotManager.start_bot(
            binance_key=binance_key,
            binance_secret=binance_secret,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat,
            openai_key=OPENAI_API_KEY,
            db=db
        )
        bot_state.running = success
        return {"ok": success, "message": "Bot started" if success else "Failed to start"}
    except Exception as e:
        logger.error(f"Start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trading/stop")
async def stop_trading_bot():
    """Stop the trading bot"""
    try:
        from trading_bot import TradingBotManager
        success = await TradingBotManager.stop_bot()
        bot_state.running = False
        return {"ok": success, "message": "Bot stopped"}
    except Exception as e:
        logger.error(f"Stop error: {e}")
        return {"ok": False, "message": str(e)}


@app.get("/api/trading/status")
async def get_trading_status():
    """Get trading bot status"""
    try:
        from trading_bot import TradingBotManager
        return TradingBotManager.get_status()
    except:
        return {"running": False}


@app.get("/api/trading/trades")
async def get_trades(limit: int = 50):
    """Get recent trades"""
    trades = await db.trades.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit).to_list(limit)
    for t in trades:
        if 'timestamp' in t and isinstance(t['timestamp'], datetime):
            t['timestamp'] = t['timestamp'].isoformat()
    return trades


@app.get("/api/trading/reports")
async def get_reports(limit: int = 30):
    """Get daily reports"""
    reports = await db.daily_reports.find({}, {"_id": 0}).sort("date", -1).limit(limit).to_list(limit)
    return reports


# ============== Startup ==============

@app.on_event("startup")
async def startup():
    logger.info("🚀 AI Trading Bot API started")
    logger.info(f"📊 MongoDB: {MONGO_URL[:30]}...")
    logger.info(f"🔑 OpenAI: configured")
    logger.info(f"📈 Binance: configured")
    logger.info(f"📱 Telegram: configured")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down...")
    client.close()
