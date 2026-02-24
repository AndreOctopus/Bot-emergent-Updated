# 🤖 AI Crypto Trading Bot - Deployment Guide

## 📋 Вимоги

- Python 3.11+
- Node.js 18+
- MongoDB
- VPS сервер БЕЗ обмежень Binance (Європа, Азія)

## 🔑 Необхідні API ключі

```env
# Backend (.env)
MONGO_URL=mongodb://localhost:27017
DB_NAME=trading_bot
EMERGENT_API_KEY=your_emergent_llm_key

# Binance Futures
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## 📁 Структура проекту

```
trading-bot/
├── backend/
│   ├── server.py              # FastAPI сервер
│   ├── trading_bot.py         # Головна логіка бота
│   ├── backtest_synthetic.py  # Бектест
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   ├── pages/
│   │   │   └── TradingDashboard.jsx
│   │   └── components/ui/
│   └── package.json
└── docker-compose.yml
```

## 🚀 Швидкий старт

### 1. Клонування та налаштування

```bash
# Створити директорію
mkdir trading-bot && cd trading-bot

# Скопіювати файли (з цього репо)
# backend/server.py
# backend/trading_bot.py
# frontend/src/...
```

### 2. Backend Setup

```bash
cd backend

# Створити virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# або: venv\Scripts\activate  # Windows

# Встановити залежності
pip install -r requirements.txt

# Створити .env файл
cp .env.example .env
# Заповнити ключі в .env

# Запустити сервер
uvicorn server:app --host 0.0.0.0 --port 8001
```

### 3. Frontend Setup

```bash
cd frontend

# Встановити залежності
yarn install

# Створити .env
echo "REACT_APP_BACKEND_URL=http://localhost:8001" > .env

# Запустити dev server
yarn start
```

### 4. MongoDB

```bash
# Docker варіант
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Або встановити локально
# https://www.mongodb.com/docs/manual/installation/
```

## 🐳 Docker Deployment

### docker-compose.yml

```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

  backend:
    build: ./backend
    ports:
      - "8001:8001"
    environment:
      - MONGO_URL=mongodb://mongodb:27017
      - DB_NAME=trading_bot
      - EMERGENT_API_KEY=${EMERGENT_API_KEY}
    depends_on:
      - mongodb

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_BACKEND_URL=http://backend:8001

volumes:
  mongo_data:
```

### Запуск

```bash
docker-compose up -d
```

## ⚙️ Конфігурація стратегії

Параметри в `trading_bot.py`:

```python
# Conservative Strategy Parameters
DEFAULT_LEVERAGE = 3          # Leverage
DEFAULT_STOP_LOSS_PERCENT = 1.0    # Stop Loss %
DEFAULT_TAKE_PROFIT_PERCENT = 1.5  # Take Profit %
DEFAULT_RISK_PER_TRADE = 3.0       # % балансу на угоду
MIN_SIGNAL_CONFIDENCE = 80         # Мін. впевненість AI
daily_target_percent = 2.5         # Денна ціль %
max_positions = 2                  # Макс позицій
```

## 📊 API Endpoints

| Endpoint | Method | Опис |
|----------|--------|------|
| `/api/trading/start` | POST | Запустити бота |
| `/api/trading/stop` | POST | Зупинити бота |
| `/api/trading/status` | GET | Статус бота |
| `/api/trading/trades` | GET | Історія угод |
| `/api/trading/reports` | GET | Добові звіти |

## 🔒 Безпека

1. **Ніколи не коміть API ключі** в репозиторій
2. Використовуйте `.env` файли
3. Binance API: увімкніть лише Futures trading
4. Обмежте IP для API ключів

## 🌍 Рекомендовані VPS провайдери

(без обмежень Binance):
- Hetzner (Німеччина)
- OVH (Франція)
- Vultr (Сінгапур/Японія)
- DigitalOcean (Сінгапур)

## 📞 Telegram Bot Setup

1. Створити бота через @BotFather
2. Отримати токен
3. Створити канал/групу
4. Додати бота в канал
5. Отримати chat_id через API

## ⚠️ Disclaimer

Торгівля криптовалютами несе високий ризик втрати коштів.
Це не фінансова порада. Торгуйте лише тими коштами,
які готові втратити.

---

📧 Support: Emergent Platform
🔗 Dashboard: /trading
