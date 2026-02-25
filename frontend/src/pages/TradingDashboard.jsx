import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { toast } from 'sonner';
import { 
  Bot, Play, Square, TrendingUp, TrendingDown, 
  DollarSign, Activity, BarChart3, Clock, Target,
  Wallet, Zap, AlertTriangle, CheckCircle2, XCircle
} from 'lucide-react';
import axios from 'axios';

// Backend URL - use environment variable or relative path for Railway
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || '';

const TradingDashboard = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState(null);
  const [trades, setTrades] = useState([]);
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(false);
  
  // Bot configuration - hardcoded defaults (can be overridden)
  const [config, setConfig] = useState({
    binance_api_key: 'cuCbg2Yz9Co9lBCSUFHGuGqfK44s69NQOKRsbgJD97KzUIv6KsQCC0u6t4zc1a3I',
    binance_secret_key: 'Vz8VPUugXwaAfIFh6NmIb6OGu8cOIh1OZuoEgsebIl08UPyAn6ErUYAjfLmV01Xa',
    telegram_token: '8144215710:AAEX_U3V2HQhJ5AhVpFJbt5CRm2dA5yDiHg',
    telegram_chat_id: '-1003844330472'
  });
  
  const [showConfig, setShowConfig] = useState(true);

  // Fetch bot status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/trading/status`, {
        withCredentials: true
      });
      setStatus(response.data);
      setIsRunning(response.data.running);
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  }, []);

  // Fetch recent trades
  const fetchTrades = useCallback(async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/trading/trades?limit=20`, {
        withCredentials: true
      });
      setTrades(response.data);
    } catch (error) {
      console.error('Failed to fetch trades:', error);
    }
  }, []);

  // Fetch daily reports
  const fetchReports = useCallback(async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/trading/reports?limit=7`, {
        withCredentials: true
      });
      setReports(response.data);
    } catch (error) {
      console.error('Failed to fetch reports:', error);
    }
  }, []);

  // Initial load and polling
  useEffect(() => {
    fetchStatus();
    fetchTrades();
    fetchReports();
    
    const interval = setInterval(() => {
      fetchStatus();
      if (isRunning) {
        fetchTrades();
      }
    }, 10000);
    
    return () => clearInterval(interval);
  }, [fetchStatus, fetchTrades, fetchReports, isRunning]);

  // Start bot
  const startBot = async () => {
    if (!config.binance_api_key || !config.binance_secret_key || 
        !config.telegram_token || !config.telegram_chat_id) {
      toast.error('Please fill in all configuration fields');
      return;
    }
    
    setLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/trading/start`, config, {
        withCredentials: true
      });
      
      if (response.data.ok) {
        toast.success('Trading bot started! 🚀');
        setIsRunning(true);
        setShowConfig(false);
        fetchStatus();
      } else {
        toast.error(response.data.message);
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to start bot');
    } finally {
      setLoading(false);
    }
  };

  // Stop bot
  const stopBot = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/trading/stop`, {}, {
        withCredentials: true
      });
      
      if (response.data.ok) {
        toast.success('Trading bot stopped');
        setIsRunning(false);
        fetchStatus();
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to stop bot');
    } finally {
      setLoading(false);
    }
  };

  // Calculate stats
  const todayPnL = status?.daily_start_balance && status?.current_balance 
    ? ((status.current_balance - status.daily_start_balance) / status.daily_start_balance * 100)
    : 0;
  
  const winRate = status?.trades_today > 0 
    ? (status.winning_trades / status.trades_today * 100)
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-gradient-to-br from-amber-500 to-orange-600 rounded-xl">
              <Bot className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
                AI Crypto Scalper
              </h1>
              <p className="text-slate-400">Conservative Mode • +2.5% Daily Target</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <Badge 
              data-testid="bot-status-badge"
              className={`px-4 py-2 text-sm ${
                isRunning 
                  ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50' 
                  : 'bg-slate-700/50 text-slate-400 border-slate-600/50'
              }`}
            >
              <Activity className={`w-4 h-4 mr-2 ${isRunning ? 'animate-pulse' : ''}`} />
              {isRunning ? 'RUNNING' : 'STOPPED'}
            </Badge>
            
            {isRunning ? (
              <Button 
                data-testid="stop-bot-btn"
                onClick={stopBot} 
                disabled={loading}
                className="bg-red-600 hover:bg-red-700"
              >
                <Square className="w-4 h-4 mr-2" />
                Stop Bot
              </Button>
            ) : (
              <Button 
                data-testid="start-bot-btn"
                onClick={() => setShowConfig(true)}
                className="bg-gradient-to-r from-amber-500 to-orange-600 hover:from-amber-600 hover:to-orange-700"
              >
                <Play className="w-4 h-4 mr-2" />
                Start Bot
              </Button>
            )}
          </div>
        </div>

        {/* Configuration Panel */}
        {showConfig && !isRunning && (
          <Card data-testid="config-panel" className="mb-8 bg-slate-900/50 border-slate-800 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-xl flex items-center gap-2">
                <Zap className="w-5 h-5 text-amber-500" />
                Bot Configuration
              </CardTitle>
              <CardDescription>Enter your API keys to start trading</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="binance_key">Binance API Key</Label>
                  <Input
                    id="binance_key"
                    data-testid="binance-api-key-input"
                    type="password"
                    placeholder="Enter your Binance API key"
                    value={config.binance_api_key}
                    onChange={(e) => setConfig({...config, binance_api_key: e.target.value})}
                    className="bg-slate-800 border-slate-700"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="binance_secret">Binance Secret Key</Label>
                  <Input
                    id="binance_secret"
                    data-testid="binance-secret-key-input"
                    type="password"
                    placeholder="Enter your Binance Secret key"
                    value={config.binance_secret_key}
                    onChange={(e) => setConfig({...config, binance_secret_key: e.target.value})}
                    className="bg-slate-800 border-slate-700"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="telegram_token">Telegram Bot Token</Label>
                  <Input
                    id="telegram_token"
                    data-testid="telegram-token-input"
                    type="password"
                    placeholder="Enter Telegram bot token"
                    value={config.telegram_token}
                    onChange={(e) => setConfig({...config, telegram_token: e.target.value})}
                    className="bg-slate-800 border-slate-700"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="telegram_chat">Telegram Chat ID</Label>
                  <Input
                    id="telegram_chat"
                    data-testid="telegram-chat-input"
                    placeholder="Enter Telegram chat/channel ID"
                    value={config.telegram_chat_id}
                    onChange={(e) => setConfig({...config, telegram_chat_id: e.target.value})}
                    className="bg-slate-800 border-slate-700"
                  />
                </div>
              </div>
              
              <div className="flex justify-end gap-4">
                <Button 
                  variant="outline" 
                  onClick={() => setShowConfig(false)}
                  className="border-slate-700"
                >
                  Cancel
                </Button>
                <Button 
                  data-testid="confirm-start-btn"
                  onClick={startBot}
                  disabled={loading}
                  className="bg-gradient-to-r from-amber-500 to-orange-600"
                >
                  {loading ? 'Starting...' : 'Start Trading'}
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <Card data-testid="balance-card" className="bg-slate-900/50 border-slate-800 backdrop-blur">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Balance</p>
                  <p className="text-2xl font-bold text-white">
                    ${(status?.daily_start_balance || 0).toLocaleString(undefined, {maximumFractionDigits: 2})}
                  </p>
                </div>
                <div className="p-3 bg-emerald-500/20 rounded-lg">
                  <Wallet className="w-6 h-6 text-emerald-400" />
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card data-testid="pnl-card" className="bg-slate-900/50 border-slate-800 backdrop-blur">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Today's P&L</p>
                  <p className={`text-2xl font-bold ${todayPnL >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {todayPnL >= 0 ? '+' : ''}{todayPnL.toFixed(2)}%
                  </p>
                </div>
                <div className={`p-3 rounded-lg ${todayPnL >= 0 ? 'bg-emerald-500/20' : 'bg-red-500/20'}`}>
                  {todayPnL >= 0 ? (
                    <TrendingUp className="w-6 h-6 text-emerald-400" />
                  ) : (
                    <TrendingDown className="w-6 h-6 text-red-400" />
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card data-testid="trades-card" className="bg-slate-900/50 border-slate-800 backdrop-blur">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Trades Today</p>
                  <p className="text-2xl font-bold text-white">
                    {status?.trades_today || 0}
                  </p>
                  <p className="text-xs text-slate-500">
                    {status?.winning_trades || 0}W / {status?.losing_trades || 0}L
                  </p>
                </div>
                <div className="p-3 bg-blue-500/20 rounded-lg">
                  <BarChart3 className="w-6 h-6 text-blue-400" />
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card data-testid="winrate-card" className="bg-slate-900/50 border-slate-800 backdrop-blur">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">Win Rate</p>
                  <p className="text-2xl font-bold text-white">
                    {winRate.toFixed(1)}%
                  </p>
                </div>
                <div className="p-3 bg-purple-500/20 rounded-lg">
                  <Target className="w-6 h-6 text-purple-400" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Target Progress */}
        <Card data-testid="target-progress" className="mb-8 bg-slate-900/50 border-slate-800 backdrop-blur">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold">Daily Target Progress</h3>
                <p className="text-sm text-slate-400">Target: +2.5% daily profit (Conservative)</p>
              </div>
              <Badge className={`${todayPnL >= 2.5 ? 'bg-emerald-500' : 'bg-amber-500/50'}`}>
                {todayPnL >= 2.5 ? '🎉 TARGET REACHED!' : `${Math.min(todayPnL / 2.5 * 100, 100).toFixed(0)}%`}
              </Badge>
            </div>
            <div className="h-4 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-amber-500 to-emerald-500 transition-all duration-500"
                style={{ width: `${Math.min(Math.max(todayPnL / 2.5 * 100, 0), 100)}%` }}
              />
            </div>
            <div className="flex justify-between mt-2 text-xs text-slate-500">
              <span>0%</span>
              <span>1.25%</span>
              <span>2.5%</span>
            </div>
          </CardContent>
        </Card>

        {/* Tabs: Trades & Reports */}
        <Tabs defaultValue="trades" className="space-y-4">
          <TabsList className="bg-slate-900/50 border border-slate-800">
            <TabsTrigger value="trades" data-testid="trades-tab">Recent Trades</TabsTrigger>
            <TabsTrigger value="reports" data-testid="reports-tab">Daily Reports</TabsTrigger>
          </TabsList>
          
          <TabsContent value="trades">
            <Card className="bg-slate-900/50 border-slate-800 backdrop-blur">
              <CardHeader>
                <CardTitle className="text-lg">Recent Trades</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[400px]">
                  {trades.length === 0 ? (
                    <div className="text-center text-slate-500 py-8">
                      No trades yet. Start the bot to begin trading.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {trades.map((trade, index) => (
                        <div 
                          key={index}
                          data-testid={`trade-item-${index}`}
                          className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg"
                        >
                          <div className="flex items-center gap-4">
                            <div className={`p-2 rounded-lg ${
                              trade.side === 'BUY' ? 'bg-emerald-500/20' : 'bg-red-500/20'
                            }`}>
                              {trade.side === 'BUY' ? (
                                <TrendingUp className={`w-5 h-5 text-emerald-400`} />
                              ) : (
                                <TrendingDown className={`w-5 h-5 text-red-400`} />
                              )}
                            </div>
                            <div>
                              <p className="font-semibold">{trade.symbol}</p>
                              <p className="text-sm text-slate-400">
                                {trade.side} @ ${trade.entry_price?.toLocaleString()} | {trade.leverage}x
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <Badge className={`${
                              trade.action === 'OPEN' ? 'bg-blue-500/20 text-blue-400' : 'bg-slate-700'
                            }`}>
                              {trade.action}
                            </Badge>
                            <p className="text-xs text-slate-500 mt-1">
                              {new Date(trade.timestamp).toLocaleString()}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="reports">
            <Card className="bg-slate-900/50 border-slate-800 backdrop-blur">
              <CardHeader>
                <CardTitle className="text-lg">Daily Performance Reports</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[400px]">
                  {reports.length === 0 ? (
                    <div className="text-center text-slate-500 py-8">
                      No reports yet. Reports are generated at end of each trading day.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {reports.map((report, index) => (
                        <div 
                          key={index}
                          data-testid={`report-item-${index}`}
                          className="p-4 bg-slate-800/50 rounded-lg"
                        >
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-3">
                              <Clock className="w-5 h-5 text-slate-400" />
                              <span className="font-semibold">{report.date}</span>
                            </div>
                            {report.target_reached ? (
                              <Badge className="bg-emerald-500/20 text-emerald-400">
                                <CheckCircle2 className="w-4 h-4 mr-1" />
                                Target Reached
                              </Badge>
                            ) : (
                              <Badge className="bg-slate-700 text-slate-400">
                                <XCircle className="w-4 h-4 mr-1" />
                                Target Missed
                              </Badge>
                            )}
                          </div>
                          <div className="grid grid-cols-4 gap-4 text-sm">
                            <div>
                              <p className="text-slate-500">P&L</p>
                              <p className={`font-semibold ${
                                report.profit_loss >= 0 ? 'text-emerald-400' : 'text-red-400'
                              }`}>
                                ${report.profit_loss?.toFixed(2)} ({report.profit_loss_percent?.toFixed(2)}%)
                              </p>
                            </div>
                            <div>
                              <p className="text-slate-500">Trades</p>
                              <p className="font-semibold">{report.trades_count}</p>
                            </div>
                            <div>
                              <p className="text-slate-500">Wins</p>
                              <p className="font-semibold text-emerald-400">{report.winning_trades}</p>
                            </div>
                            <div>
                              <p className="text-slate-500">Losses</p>
                              <p className="font-semibold text-red-400">{report.losing_trades}</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Risk Warning */}
        <Card className="mt-8 bg-amber-900/20 border-amber-500/30">
          <CardContent className="pt-6">
            <div className="flex items-start gap-4">
              <AlertTriangle className="w-6 h-6 text-amber-500 flex-shrink-0 mt-1" />
              <div>
                <h3 className="font-semibold text-amber-400 mb-2">Risk Warning</h3>
                <p className="text-sm text-amber-200/70">
                  Cryptocurrency futures trading involves substantial risk of loss. The AI trading bot uses leverage 
                  which can amplify both gains and losses. Past performance does not guarantee future results. 
                  Only trade with funds you can afford to lose. This is not financial advice.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Strategy Info */}
        <Card className="mt-4 bg-emerald-900/20 border-emerald-500/30">
          <CardContent className="pt-6">
            <h3 className="font-semibold text-emerald-400 mb-3">📊 Conservative Strategy Parameters</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-slate-500">Daily Target</p>
                <p className="text-emerald-400 font-semibold">+2.5%</p>
              </div>
              <div>
                <p className="text-slate-500">Leverage</p>
                <p className="text-emerald-400 font-semibold">3x</p>
              </div>
              <div>
                <p className="text-slate-500">Stop Loss</p>
                <p className="text-red-400 font-semibold">1%</p>
              </div>
              <div>
                <p className="text-slate-500">Take Profit</p>
                <p className="text-emerald-400 font-semibold">1.5%</p>
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-3">
              🎯 Projected monthly return: +50-75% | Trading 5 pairs: BTC, ETH, BNB, SOL, XRP
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default TradingDashboard;
