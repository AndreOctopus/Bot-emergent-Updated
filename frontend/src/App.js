import React from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";  // Видали useLocation, Navigate — не потрібні
import TradingDashboard from "@/pages/TradingDashboard";  // Додай цей імпорт
import { Toaster } from "@/components/ui/sonner";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="*" element={<TradingDashboard />} />  // Все веде на дашборд, ігноруй авторизацію
      </Routes>
      <Toaster />
    </BrowserRouter>
  );
}

export default App;
