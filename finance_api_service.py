"""
Finance API Service for Real-Time Financial Data
Handles live API calls for currency rates, gold prices, market data, and tax information
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import time

# Optional imports for enhanced functionality
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not available. Stock market data will be limited.")

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("⚠️ beautifulsoup4 not available. Web scraping features will be limited.")


class FinanceAPIService:
    """Service layer for fetching real-time finance data."""

    def __init__(self):
        # API endpoints and configuration
        self.currency_api = "https://www.goldapi.io/goldapi-3n3d8smhn210qq-io/XAU/USD"
        self.backup_currency_api = "https://api.fxratesapi.com/latest?base=INR"
        
        # Cache for API responses (to avoid rate limiting)
        self._cache = {}
        self._cache_duration = 300  # 5 minutes cache
        
        # Indian market symbols for yfinance
        self.indian_indices = {
            "NIFTY": "^NSEI",
            "SENSEX": "^BSESN",
            "BANKNIFTY": "^NSEBANK",
            "NIFTYIT": "^CNXIT"
        }
        
        print("💹 Finance API Service initialized")

    def _get_cached_or_fetch(self, cache_key: str, fetch_function, *args, **kwargs):
        """Generic caching mechanism for API calls"""
        current_time = time.time()
        
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if current_time - timestamp < self._cache_duration:
                return cached_data
        
        # Fetch new data
        try:
            data = fetch_function(*args, **kwargs)
            self._cache[cache_key] = (data, current_time)
            return data
        except Exception as e:
            # Return cached data if available, even if expired
            if cache_key in self._cache:
                print(f"⚠️ API call failed, using cached data: {e}")
                return self._cache[cache_key][0]
            raise e

    def get_currency_rate(self, target: str = "USD") -> Dict[str, Any]:
        """Fetch INR to other currency rate with fallback APIs."""
        
        def _fetch_currency_rate():
            try:
                # Primary API
                response = requests.get(self.currency_api, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                rate = data["rates"].get(target.upper())
                if rate is None:
                    raise ValueError(f"Currency {target} not found")
                
                return {
                    "success": True,
                    "base": "INR",
                    "target": target.upper(),
                    "rate": round(rate, 4),
                    "date": data.get("date", str(datetime.now().date())),
                    "timestamp": datetime.now().isoformat(),
                    "source": "exchangerate-api.com"
                }
                
            except Exception as e:
                print(f"Primary currency API failed: {e}")
                
                # Fallback: Use backup API
                try:
                    backup_response = requests.get(self.backup_currency_api, timeout=10)
                    backup_data = backup_response.json()
                    
                    rate = backup_data["rates"].get(target.upper())
                    if rate:
                        return {
                            "success": True,
                            "base": "INR",
                            "target": target.upper(),
                            "rate": round(rate, 4),
                            "date": str(datetime.now().date()),
                            "timestamp": datetime.now().isoformat(),
                            "source": "fxratesapi.com (backup)"
                        }
                except Exception as backup_error:
                    print(f"Backup currency API also failed: {backup_error}")
                
                # Final fallback: Static approximate rates
                static_rates = {
                    "USD": 83.25, "EUR": 90.50, "GBP": 105.75, "JPY": 0.55,
                    "AUD": 54.20, "CAD": 61.30, "SGD": 61.80, "AED": 22.65
                }
                
                if target.upper() in static_rates:
                    return {
                        "success": False,
                        "base": "INR",
                        "target": target.upper(),
                        "rate": static_rates[target.upper()],
                        "date": str(datetime.now().date()),
                        "timestamp": datetime.now().isoformat(),
                        "source": "static_fallback",
                        "note": "Live data unavailable, showing approximate rate"
                    }
                
                return {
                    "success": False,
                    "error": f"Unable to fetch rate for {target}",
                    "timestamp": datetime.now().isoformat()
                }
        
        return self._get_cached_or_fetch(f"currency_{target}", _fetch_currency_rate)

    def get_gold_price_inr(self) -> Dict[str, Any]:
        """Return hardcoded gold price (no API calls)."""
        try:
            # Hardcoded values — use these until live API integration is fixed
            price_per_gram = 12342.00
            price_per_10g = 123420.00

            return {
                "success": True,
                "gold_inr_per_gram": price_per_gram,
                "gold_inr_per_10g": price_per_10g,
                "date": str(datetime.now().date()),
                "timestamp": datetime.now().isoformat(),
                "source": "hardcoded_static",
                "note": "Static gold price for testing; live API temporarily disabled"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error returning hardcoded gold price: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        
        def _fetch_gold_price():
            try:
                # Try multiple gold price APIs
                apis_to_try = [
                    "https://api.metalpriceapi.com/v1/latest?api_key=demo&base=INR&currencies=XAU",
                    
                ]
                
                for api_url in apis_to_try:
                    try:
                        response = requests.get(api_url, timeout=10)
                        data = response.json()
                        
                        if "rates" in data and "XAU" in data["rates"]:
                            # XAU is troy ounce, convert to gram
                            xau_rate = data["rates"]["XAU"]
                            price_per_gram = (1 / xau_rate) / 31.1035  # 1 troy ounce = 31.1035 grams
                            
                            return {
                                "success": True,
                                "gold_inr_per_gram": round(price_per_gram, 2),
                                "gold_inr_per_10g": round(price_per_gram * 10, 2),
                                "date": str(datetime.now().date()),
                                "timestamp": datetime.now().isoformat(),
                                "source": "metalpriceapi.com"
                            }
                    except Exception as api_error:
                        print(f"Gold API {api_url} failed: {api_error}")
                        continue
                
                # Fallback: Approximate current gold price
                return {
                    "success": False,
                    "gold_inr_per_gram": 6850.00,  # Approximate current rate
                    "gold_inr_per_10g": 68500.00,
                    "date": str(datetime.now().date()),
                    "timestamp": datetime.now().isoformat(),
                    "source": "static_fallback",
                    "note": "Live data unavailable, showing approximate price"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Unable to fetch gold price: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
        
        return self._get_cached_or_fetch("gold_price", _fetch_gold_price)

    def get_indian_market_data(self) -> Dict[str, Any]:
        """Fetch Indian stock market indices (Nifty, Sensex, etc.)"""
        
        def _fetch_market_data():
            market_data = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "indices": {},
                "source": "yfinance" if YFINANCE_AVAILABLE else "static_fallback"
            }
            
            if YFINANCE_AVAILABLE:
                try:
                    for name, symbol in self.indian_indices.items():
                        try:
                            ticker = yf.Ticker(symbol)
                            info = ticker.info
                            hist = ticker.history(period="2d")
                            
                            if not hist.empty:
                                current_price = hist['Close'].iloc[-1]
                                previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                                change = current_price - previous_price
                                change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
                                
                                market_data["indices"][name] = {
                                    "current_price": round(current_price, 2),
                                    "previous_close": round(previous_price, 2),
                                    "change": round(change, 2),
                                    "change_percent": round(change_percent, 2),
                                    "symbol": symbol,
                                    "name": info.get("longName", name)
                                }
                        except Exception as ticker_error:
                            print(f"Failed to fetch {name}: {ticker_error}")
                            continue
                            
                except Exception as e:
                    print(f"YFinance error: {e}")
                    market_data["success"] = False
            
            # Fallback data if yfinance fails or unavailable
            if not market_data["indices"] or not YFINANCE_AVAILABLE:
                market_data["indices"] = {
                    "NIFTY": {
                        "current_price": 24500.00,
                        "previous_close": 24450.00,
                        "change": 50.00,
                        "change_percent": 0.20,
                        "symbol": "^NSEI",
                        "name": "NIFTY 50"
                    },
                    "SENSEX": {
                        "current_price": 80500.00,
                        "previous_close": 80300.00,
                        "change": 200.00,
                        "change_percent": 0.25,
                        "symbol": "^BSESN", 
                        "name": "BSE SENSEX"
                    }
                }
                market_data["source"] = "static_fallback"
                market_data["note"] = "Live data unavailable, showing approximate values"
            
            return market_data
        
        return self._get_cached_or_fetch("market_data", _fetch_market_data)

    def get_tax_slabs(self, year: str = "2024-25") -> Dict[str, Any]:
        """Fetch latest Indian income tax slabs."""
        return {
            "success": True,
            "year": year,
            "timestamp": datetime.now().isoformat(),
            "old_regime": [
                {"income_range": "Up to ₹2.5 Lakh", "tax_rate": "0%", "tax_amount": "₹0"},
                {"income_range": "₹2.5 - ₹5 Lakh", "tax_rate": "5%", "tax_amount": "₹0 - ₹12,500"},
                {"income_range": "₹5 - ₹10 Lakh", "tax_rate": "20%", "tax_amount": "₹12,500 - ₹1,12,500"},
                {"income_range": "Above ₹10 Lakh", "tax_rate": "30%", "tax_amount": "₹1,12,500 + 30% of excess"}
            ],
            "new_regime": [
                {"income_range": "Up to ₹3 Lakh", "tax_rate": "0%", "tax_amount": "₹0"},
                {"income_range": "₹3 - ₹6 Lakh", "tax_rate": "5%", "tax_amount": "₹0 - ₹15,000"},
                {"income_range": "₹6 - ₹9 Lakh", "tax_rate": "10%", "tax_amount": "₹15,000 - ₹45,000"},
                {"income_range": "₹9 - ₹12 Lakh", "tax_rate": "15%", "tax_amount": "₹45,000 - ₹90,000"},
                {"income_range": "₹12 - ₹15 Lakh", "tax_rate": "20%", "tax_amount": "₹90,000 - ₹1,50,000"},
                {"income_range": "Above ₹15 Lakh", "tax_rate": "30%", "tax_amount": "₹1,50,000 + 30% of excess"}
            ],
            "standard_deduction": "₹50,000 (for salaried individuals)",
            "cess": "4% Health and Education Cess on total tax",
            "note": "Rates for Assessment Year 2024-25. Consult a tax advisor for personalized advice."
        }

    def get_mutual_fund_nav(self, scheme_code: str = "120503") -> Dict[str, Any]:
        """Fetch mutual fund NAV (Net Asset Value) - example with SBI Bluechip Fund"""
        
        def _fetch_nav():
            try:
                # Using AMFI API for mutual fund data
                api_url = f"https://api.mfapi.in/mf/{scheme_code}"
                response = requests.get(api_url, timeout=10)
                data = response.json()
                
                if data and "data" in data and data["data"]:
                    latest = data["data"][0]
                    return {
                        "success": True,
                        "scheme_name": data.get("meta", {}).get("scheme_name", "Unknown Fund"),
                        "scheme_code": scheme_code,
                        "nav": float(latest["nav"]),
                        "date": latest["date"],
                        "timestamp": datetime.now().isoformat(),
                        "source": "mfapi.in"
                    }
                else:
                    raise ValueError("No data available")
                    
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Unable to fetch mutual fund data: {str(e)}",
                    "scheme_code": scheme_code,
                    "timestamp": datetime.now().isoformat()
                }
        
        return self._get_cached_or_fetch(f"mf_nav_{scheme_code}", _fetch_nav)

    def get_crypto_prices(self, symbols: List[str] = ["bitcoin", "ethereum"]) -> Dict[str, Any]:
        """Fetch cryptocurrency prices in INR"""
        
        def _fetch_crypto():
            try:
                # Using CoinGecko API (free tier)
                symbols_str = ",".join(symbols)
                api_url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbols_str}&vs_currencies=inr&include_24hr_change=true"
                
                response = requests.get(api_url, timeout=10)
                data = response.json()
                
                crypto_data = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "prices": {},
                    "source": "coingecko.com"
                }
                
                for symbol in symbols:
                    if symbol in data:
                        crypto_data["prices"][symbol] = {
                            "price_inr": data[symbol]["inr"],
                            "change_24h": data[symbol].get("inr_24h_change", 0),
                            "symbol": symbol.upper()
                        }
                
                return crypto_data
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Unable to fetch crypto prices: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
        
        return self._get_cached_or_fetch("crypto_prices", _fetch_crypto)

    def get_financial_summary(self) -> Dict[str, Any]:
        """Get a comprehensive financial market summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "summary": {}
        }
        
        try:
            # Get all data
            usd_rate = self.get_currency_rate("USD")
            gold_price = self.get_gold_price_inr()
            market_data = self.get_indian_market_data()
            
            summary["summary"] = {
                "currency": {
                    "usd_inr": usd_rate.get("rate", "N/A"),
                    "status": "live" if usd_rate.get("success") else "approximate"
                },
                "gold": {
                    "price_per_gram": gold_price.get("gold_inr_per_gram", "N/A"),
                    "status": "live" if gold_price.get("success") else "approximate"
                },
                "market": {
                    "nifty": market_data.get("indices", {}).get("NIFTY", {}).get("current_price", "N/A"),
                    "sensex": market_data.get("indices", {}).get("SENSEX", {}).get("current_price", "N/A"),
                    "status": "live" if market_data.get("success") else "approximate"
                }
            }
            
        except Exception as e:
            summary["error"] = str(e)
        
        return summary

    def clear_cache(self):
        """Clear the API response cache"""
        self._cache.clear()
        print("🗑️ Finance API cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_items": len(self._cache),
            "cache_duration_seconds": self._cache_duration,
            "cache_keys": list(self._cache.keys())
        }
