#!/usr/bin/env python3
"""
Comprehensive Qlib Feature Analysis
Analyzes all possible features and their predictive power
Outputs results to markdown report
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pickle

# Core ML libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Advanced models
try:
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    ADVANCED_MODELS = True
    print("✅ Advanced models (LightGBM, XGBoost, CatBoost) available")
except ImportError as e:
    ADVANCED_MODELS = False
    print(f"⚠️ Some advanced models not available: {e}")

# Try Qlib (optional)
try:
    import qlib
    QLIB_AVAILABLE = True
    print("✅ Qlib available")
except ImportError:
    QLIB_AVAILABLE = False
    print("⚠️ Qlib not available, using standalone analysis")

class ComprehensiveFeatureAnalyzer:
    def __init__(self, price_file: str = "prices.txt"):
        self.price_file = price_file
        self.prices = None
        self.returns = None
        self.features_df = None
        self.individual_stock_features = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def load_data(self) -> np.ndarray:
        print("📊 Loading price data...")
        
        with open(self.price_file, 'r') as f:
            lines = f.readlines()
            
        prices = []
        for line in lines:
            row = [float(x) for x in line.strip().split()]
            prices.append(row)
            
        self.prices = np.array(prices)
        self.returns = np.zeros_like(self.prices)
        self.returns[1:] = self.prices[1:] / self.prices[:-1] - 1
        
        print(f"✅ Loaded {self.prices.shape[0]} days × {self.prices.shape[1]} instruments")
        return self.prices
    
    def calculate_all_features(self, lookback: int = 120) -> pd.DataFrame:
        print(f"🔧 Calculating comprehensive feature set (lookback={lookback})...")
        
        n_days, n_instruments = self.prices.shape
        all_data = []
        
        # Calculate features for each day and instrument
        for day in range(lookback, n_days - 1):
            print(f"\rProcessing day {day}/{n_days-1}", end="", flush=True)
            
            for inst in range(n_instruments):
                hist_prices = self.prices[day-lookback:day+1, inst]
                hist_returns = self.returns[day-lookback:day+1, inst]
                current_price = hist_prices[-1]
                
                features = self._calculate_instrument_features(
                    hist_prices, hist_returns, current_price, day, inst
                )
                
                # Add cross-sectional features
                cross_features = self._calculate_cross_sectional_features(day, inst)
                features.update(cross_features)
                
                # Target: next day return
                target = self.returns[day + 1, inst]
                
                # Combine all
                row_data = {
                    'day': day,
                    'instrument': inst,
                    'price': current_price,
                    'target': target,
                    **features
                }
                
                all_data.append(row_data)
        
        print("\n✅ Feature calculation complete")
        
        self.features_df = pd.DataFrame(all_data)
        
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['day', 'instrument', 'price', 'target']]
        
        for col in feature_cols:
            self.features_df[col] = pd.to_numeric(self.features_df[col], errors='coerce')
            self.features_df[col] = self.features_df[col].fillna(0)
            self.features_df[col] = self.features_df[col].replace([np.inf, -np.inf], 0)
        
        print(f"✅ Created {len(self.features_df)} samples with {len(feature_cols)} features")
        return self.features_df
    
    def _calculate_instrument_features(self, hist_prices, hist_returns, current_price, day, inst):
        """Calculate all features for a single instrument"""
        features = {}
        
        # === BASIC PRICE FEATURES ===
        # Returns at multiple horizons
        for period in [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120]:
            if len(hist_prices) > period:
                features[f'return_{period}d'] = hist_prices[-1] / hist_prices[-period-1] - 1
                features[f'log_return_{period}d'] = np.log(hist_prices[-1] / hist_prices[-period-1])
        
        # === MOVING AVERAGES ===
        for ma_period in [3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120]:
            if len(hist_prices) >= ma_period:
                ma = np.mean(hist_prices[-ma_period:])
                features[f'ma_{ma_period}'] = ma
                features[f'price_to_ma_{ma_period}'] = current_price / ma - 1 if ma > 0 else 0
                
                # Moving average slopes
                if ma_period >= 10:
                    ma_short = np.mean(hist_prices[-ma_period//2:])
                    ma_long = np.mean(hist_prices[-ma_period:])
                    features[f'ma_slope_{ma_period}'] = (ma_short - ma_long) / ma_long if ma_long > 0 else 0
        
        # === EXPONENTIAL MOVING AVERAGES ===
        if len(hist_prices) >= 20:
            prices_series = pd.Series(hist_prices)
            for ema_span in [12, 26, 50]:
                if len(hist_prices) >= ema_span:
                    ema = prices_series.ewm(span=ema_span).mean().iloc[-1]
                    features[f'ema_{ema_span}'] = ema
                    features[f'price_to_ema_{ema_span}'] = current_price / ema - 1 if ema > 0 else 0
        
        # === VOLATILITY FEATURES ===
        for vol_period in [3, 5, 7, 10, 15, 20, 30, 45, 60, 90]:
            if len(hist_returns) >= vol_period:
                vol = np.std(hist_returns[-vol_period:])
                features[f'volatility_{vol_period}d'] = vol
                features[f'volatility_{vol_period}d_ann'] = vol * np.sqrt(252)
                
                # Volatility of volatility
                if vol_period >= 10:
                    vol_series = [np.std(hist_returns[i:i+5]) for i in range(len(hist_returns)-5)]
                    if len(vol_series) >= 5:
                        features[f'vol_of_vol_{vol_period}'] = np.std(vol_series[-vol_period//5:])
        
        # Volatility ratios
        if len(hist_returns) >= 60:
            vol_5 = np.std(hist_returns[-5:])
            vol_20 = np.std(hist_returns[-20:])
            vol_60 = np.std(hist_returns[-60:])
            
            features['vol_ratio_5_20'] = vol_5 / vol_20 if vol_20 > 0 else 1
            features['vol_ratio_20_60'] = vol_20 / vol_60 if vol_60 > 0 else 1
            features['vol_ratio_5_60'] = vol_5 / vol_60 if vol_60 > 0 else 1
        
        # === MOMENTUM INDICATORS ===
        # Rate of Change
        for roc_period in [1, 3, 5, 7, 10, 15, 20, 30, 45, 60]:
            if len(hist_prices) > roc_period:
                features[f'roc_{roc_period}'] = (hist_prices[-1] / hist_prices[-roc_period-1] - 1) * 100
        
        # RSI at multiple periods
        for rsi_period in [6, 9, 14, 21, 30]:
            if len(hist_returns) >= rsi_period:
                deltas = hist_returns[-rsi_period:]
                gains = np.mean([d for d in deltas if d > 0]) if any(d > 0 for d in deltas) else 0.001
                losses = np.mean([-d for d in deltas if d < 0]) if any(d < 0 for d in deltas) else 0.001
                rsi = 100 - (100 / (1 + gains / losses))
                features[f'rsi_{rsi_period}'] = rsi
                features[f'rsi_{rsi_period}_norm'] = (rsi - 50) / 50  # Normalized RSI
        
        # Williams %R
        for wr_period in [14, 21, 30]:
            if len(hist_prices) >= wr_period:
                high_max = np.max(hist_prices[-wr_period:])
                low_min = np.min(hist_prices[-wr_period:])
                if high_max > low_min:
                    wr = (high_max - current_price) / (high_max - low_min) * 100
                    features[f'williams_r_{wr_period}'] = wr
        
        # === TREND INDICATORS ===
        # Linear trend slopes at multiple periods
        for trend_period in [5, 10, 15, 20, 30, 45, 60]:
            if len(hist_prices) >= trend_period:
                x = np.arange(trend_period)
                y = hist_prices[-trend_period:]
                slope, intercept = np.polyfit(x, y, 1)
                features[f'trend_slope_{trend_period}'] = slope / current_price if current_price > 0 else 0
                
                # R-squared of trend
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                features[f'trend_r2_{trend_period}'] = r_squared
        
        # === BOLLINGER BANDS ===
        for bb_period in [10, 15, 20, 30]:
            if len(hist_prices) >= bb_period:
                ma = np.mean(hist_prices[-bb_period:])
                std = np.std(hist_prices[-bb_period:])
                if std > 0:
                    bb_upper = ma + 2 * std
                    bb_lower = ma - 2 * std
                    features[f'bb_upper_{bb_period}'] = bb_upper
                    features[f'bb_lower_{bb_period}'] = bb_lower
                    features[f'bb_position_{bb_period}'] = (current_price - bb_lower) / (bb_upper - bb_lower)
                    features[f'bb_width_{bb_period}'] = (bb_upper - bb_lower) / ma
                    features[f'bb_squeeze_{bb_period}'] = std / ma
        
        # === MACD INDICATORS ===
        if len(hist_prices) >= 35:
            prices_series = pd.Series(hist_prices)
            ema_12 = prices_series.ewm(span=12).mean()
            ema_26 = prices_series.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            
            features['macd'] = macd_line.iloc[-1]
            features['macd_norm'] = macd_line.iloc[-1] / current_price
            
            if len(hist_prices) >= 44:
                macd_signal = macd_line.ewm(span=9).mean()
                features['macd_signal'] = macd_signal.iloc[-1]
                features['macd_histogram'] = macd_line.iloc[-1] - macd_signal.iloc[-1]
                features['macd_hist_norm'] = features['macd_histogram'] / current_price
        
        # === STOCHASTIC OSCILLATOR ===
        for stoch_period in [5, 9, 14, 21]:
            if len(hist_prices) >= stoch_period:
                high_max = np.max(hist_prices[-stoch_period:])
                low_min = np.min(hist_prices[-stoch_period:])
                if high_max > low_min:
                    k_percent = (current_price - low_min) / (high_max - low_min) * 100
                    features[f'stoch_k_{stoch_period}'] = k_percent
                    
                    # %D (3-period SMA of %K)
                    if len(hist_prices) >= stoch_period + 2:
                        k_values = []
                        for i in range(3):
                            day_price = hist_prices[-(i+1)]
                            k_val = (day_price - low_min) / (high_max - low_min) * 100
                            k_values.append(k_val)
                        features[f'stoch_d_{stoch_period}'] = np.mean(k_values)
        
        # === COMMODITY CHANNEL INDEX ===
        for cci_period in [14, 20, 30]:
            if len(hist_prices) >= cci_period:
                typical_prices = hist_prices[-cci_period:]  # Using close as typical
                sma_tp = np.mean(typical_prices)
                mean_deviation = np.mean(np.abs(typical_prices - sma_tp))
                if mean_deviation > 0:
                    cci = (current_price - sma_tp) / (0.015 * mean_deviation)
                    features[f'cci_{cci_period}'] = cci
        
        # === AROON INDICATORS ===
        for aroon_period in [14, 25]:
            if len(hist_prices) >= aroon_period:
                high_max_idx = np.argmax(hist_prices[-aroon_period:])
                low_min_idx = np.argmin(hist_prices[-aroon_period:])
                
                aroon_up = ((aroon_period - 1 - high_max_idx) / (aroon_period - 1)) * 100
                aroon_down = ((aroon_period - 1 - low_min_idx) / (aroon_period - 1)) * 100
                
                features[f'aroon_up_{aroon_period}'] = aroon_up
                features[f'aroon_down_{aroon_period}'] = aroon_down
                features[f'aroon_oscillator_{aroon_period}'] = aroon_up - aroon_down
        
        # === STATISTICAL FEATURES ===
        for stat_period in [10, 20, 30, 60]:
            if len(hist_returns) >= stat_period:
                returns_period = hist_returns[-stat_period:]
                
                # Higher moments
                features[f'skewness_{stat_period}'] = pd.Series(returns_period).skew()
                features[f'kurtosis_{stat_period}'] = pd.Series(returns_period).kurtosis()
                
                # Percentiles
                features[f'return_p10_{stat_period}'] = np.percentile(returns_period, 10)
                features[f'return_p25_{stat_period}'] = np.percentile(returns_period, 25)
                features[f'return_p75_{stat_period}'] = np.percentile(returns_period, 75)
                features[f'return_p90_{stat_period}'] = np.percentile(returns_period, 90)
                
                # Current return position in distribution
                current_return = hist_returns[-1]
                features[f'return_percentile_{stat_period}'] = (returns_period < current_return).mean()
        
        # === PRICE POSITION FEATURES ===
        for pos_period in [10, 20, 30, 60, 90]:
            if len(hist_prices) >= pos_period:
                high_max = np.max(hist_prices[-pos_period:])
                low_min = np.min(hist_prices[-pos_period:])
                
                if high_max > low_min:
                    price_position = (current_price - low_min) / (high_max - low_min)
                    features[f'price_position_{pos_period}'] = price_position
                
                features[f'distance_from_high_{pos_period}'] = (high_max - current_price) / current_price
                features[f'distance_from_low_{pos_period}'] = (current_price - low_min) / current_price
        
        # === FIBONACCI RETRACEMENTS ===
        if len(hist_prices) >= 60:
            high_60 = np.max(hist_prices[-60:])
            low_60 = np.min(hist_prices[-60:])
            price_range = high_60 - low_60
            
            if price_range > 0:
                fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                for fib in fib_levels:
                    fib_level = high_60 - fib * price_range
                    fib_name = str(fib).replace('.', '_')
                    features[f'fib_{fib_name}_distance'] = abs(current_price - fib_level) / current_price
                    features[f'fib_{fib_name}_above'] = 1 if current_price > fib_level else 0
        
        # === ICHIMOKU CLOUD COMPONENTS ===
        if len(hist_prices) >= 52:
            # Tenkan-sen (9-period)
            if len(hist_prices) >= 9:
                tenkan_high = np.max(hist_prices[-9:])
                tenkan_low = np.min(hist_prices[-9:])
                tenkan_sen = (tenkan_high + tenkan_low) / 2
                features['ichimoku_tenkan'] = tenkan_sen
                features['price_to_tenkan'] = current_price / tenkan_sen - 1
            
            # Kijun-sen (26-period)
            if len(hist_prices) >= 26:
                kijun_high = np.max(hist_prices[-26:])
                kijun_low = np.min(hist_prices[-26:])
                kijun_sen = (kijun_high + kijun_low) / 2
                features['ichimoku_kijun'] = kijun_sen
                features['price_to_kijun'] = current_price / kijun_sen - 1
                
                # Senkou Span A (leading span A)
                senkou_a = (tenkan_sen + kijun_sen) / 2
                features['ichimoku_senkou_a'] = senkou_a
                features['price_to_senkou_a'] = current_price / senkou_a - 1
            
            # Senkou Span B (leading span B) - 52 period
            senkou_b_high = np.max(hist_prices[-52:])
            senkou_b_low = np.min(hist_prices[-52:])
            senkou_b = (senkou_b_high + senkou_b_low) / 2
            features['ichimoku_senkou_b'] = senkou_b
            features['price_to_senkou_b'] = current_price / senkou_b - 1
        
        # === PATTERN RECOGNITION ===
        if len(hist_prices) >= 20:
            # Local extrema counting
            local_maxima = 0
            local_minima = 0
            
            for i in range(1, len(hist_prices) - 1):
                if hist_prices[i] > hist_prices[i-1] and hist_prices[i] > hist_prices[i+1]:
                    local_maxima += 1
                if hist_prices[i] < hist_prices[i-1] and hist_prices[i] < hist_prices[i+1]:
                    local_minima += 1
            
            features['local_maxima_20'] = local_maxima
            features['local_minima_20'] = local_minima
            features['extrema_ratio_20'] = local_maxima / (local_minima + 1)
        
        # === VOLUME PROXY FEATURES ===
        # Using price-based proxies since no volume data
        if len(hist_returns) >= 20:
            # Activity based on return magnitude
            activity = np.mean(np.abs(hist_returns[-20:]))
            features['activity_20d'] = activity
            
            # Recent vs average activity
            recent_activity = np.mean(np.abs(hist_returns[-5:]))
            features['activity_surge'] = recent_activity / activity if activity > 0 else 1
        
        return features
    
    def _calculate_cross_sectional_features(self, day: int, inst: int) -> Dict:
        """Calculate cross-sectional (relative) features"""
        features = {}
        
        # Current cross-section data
        current_prices = self.prices[day]
        current_returns = self.returns[day]
        
        # Price rankings
        price_rank = (current_prices < current_prices[inst]).sum() / len(current_prices)
        features['price_rank'] = price_rank
        features['price_rank_centered'] = price_rank - 0.5
        
        # Return rankings
        return_rank = (current_returns < current_returns[inst]).sum() / len(current_returns)
        features['return_rank'] = return_rank
        features['return_rank_centered'] = return_rank - 0.5
        
        # Market relative metrics
        market_return = np.mean(current_returns)
        features['return_vs_market'] = current_returns[inst] - market_return
        
        market_price = np.mean(current_prices)
        features['price_vs_market'] = current_prices[inst] / market_price - 1
        
        # Beta calculation (30-day rolling)
        if day >= 30:
            inst_returns = self.returns[day-29:day+1, inst]
            market_returns = np.mean(self.returns[day-29:day+1], axis=1)
            
            if np.std(market_returns) > 0:
                correlation = np.corrcoef(inst_returns, market_returns)[0, 1]
                beta = correlation * (np.std(inst_returns) / np.std(market_returns))
                features['beta_30d'] = beta if not np.isnan(beta) else 0
                features['correlation_market_30d'] = correlation if not np.isnan(correlation) else 0
            else:
                features['beta_30d'] = 0
                features['correlation_market_30d'] = 0
        
        # Relative strength vs market at different horizons
        for rs_period in [5, 10, 20]:
            if day >= rs_period:
                inst_cumret = np.prod(1 + self.returns[day-rs_period+1:day+1, inst]) - 1
                market_cumret = np.mean([np.prod(1 + self.returns[day-rs_period+1:day+1, i]) - 1 
                                       for i in range(self.prices.shape[1])])
                features[f'relative_strength_{rs_period}d'] = inst_cumret - market_cumret
        
        # Sector proxy (based on price correlation clustering)
        if day >= 60:
            # Simple sector proxy: group by price correlation
            correlations = []
            inst_rets = self.returns[day-59:day+1, inst]
            
            for other_inst in range(self.prices.shape[1]):
                if other_inst != inst:
                    other_rets = self.returns[day-59:day+1, other_inst]
                    corr = np.corrcoef(inst_rets, other_rets)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                features['avg_correlation'] = np.mean(correlations)
                features['max_correlation'] = np.max(correlations)
                features['correlation_dispersion'] = np.std(correlations)
        
        return features
    
    def analyze_individual_stock_features(self) -> Dict[int, Dict]:
        """Analyze features for each individual stock"""
        print("📊 Analyzing individual stock features...")
        
        if self.features_df is None:
            raise ValueError("Features not calculated yet. Run calculate_all_features() first.")
        
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['day', 'instrument', 'price', 'target']]
        
        stock_analysis = {}
        
        for stock_id in range(self.prices.shape[1]):
            print(f"  Analyzing Stock {stock_id}...")
            
            # Get data for this stock only
            stock_data = self.features_df[self.features_df['instrument'] == stock_id].copy()
            
            if len(stock_data) < 30:  # Skip stocks with insufficient data
                continue
            
            clean_data = stock_data.dropna(subset=['target'])
            X = clean_data[feature_cols].fillna(0).values
            y = clean_data['target'].values
            
            # Remove outliers
            y_p95 = np.percentile(y, 95)
            y_p5 = np.percentile(y, 5)
            mask = (y >= y_p5) & (y <= y_p95)
            X, y = X[mask], y[mask]
            
            if len(X) < 20:  # Skip if too few samples after cleaning
                continue
            
            stock_features = {}
            
            # Calculate feature correlation for this stock
            for i, col in enumerate(feature_cols):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                stock_features[col] = abs(corr) if not np.isnan(corr) else 0
            
            # Sort all features by correlation
            sorted_features = sorted(stock_features.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate stock-specific metrics
            stock_stats = {
                'total_samples': len(clean_data),
                'mean_return': np.mean(y),
                'return_volatility': np.std(y),
                'all_feature_correlations': stock_features,
                'ranked_features': sorted_features,  # All features ranked
                'feature_statistics': {}
            }
            
            # Calculate feature statistics for this stock
            for i, col in enumerate(feature_cols):
                feature_values = X[:, i]
                stock_stats['feature_statistics'][col] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'min': np.min(feature_values),
                    'max': np.max(feature_values),
                    'median': np.median(feature_values),
                    'correlation_with_target': stock_features[col]
                }
            
            # Ridge regression analysis for this stock
            stock_stats['ridge_analysis'] = self._analyze_ridge_for_stock(X, y, feature_cols)
            
            stock_analysis[stock_id] = stock_stats
        
        self.individual_stock_features = stock_analysis
        print(f"✅ Analyzed {len(stock_analysis)} stocks individually with all features")
        
        return stock_analysis
    
    def _analyze_ridge_for_stock(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> Dict:
        """Perform Ridge regression analysis for individual stock"""
        from sklearn.model_selection import train_test_split
        
        if len(X) < 50:
            return {'error': 'Insufficient data for Ridge analysis'}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Test different Ridge alphas
            alphas = [0.1, 1.0, 10.0, 100.0]
            best_alpha = 1.0
            best_ic = 0
            
            for alpha in alphas:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_train_scaled, y_train)
                y_pred = ridge.predict(X_test_scaled)
                
                ic = np.corrcoef(y_test, y_pred)[0, 1]
                if not np.isnan(ic) and abs(ic) > abs(best_ic):
                    best_ic = ic
                    best_alpha = alpha
            
            # Final model with best alpha
            ridge = Ridge(alpha=best_alpha)
            ridge.fit(X_train_scaled, y_train)
            y_pred = ridge.predict(X_test_scaled)
            
            final_ic = np.corrcoef(y_test, y_pred)[0, 1]
            if np.isnan(final_ic):
                final_ic = 0
            
            # Feature importance from Ridge coefficients
            feature_importance = {}
            for i, feature in enumerate(feature_cols):
                feature_importance[feature] = abs(ridge.coef_[i])
            
            # Sort features by Ridge importance
            ridge_ranked = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'best_alpha': best_alpha,
                'information_coefficient': final_ic,
                'feature_importance': feature_importance,
                'ridge_ranked_features': ridge_ranked,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'predictability_score': abs(final_ic)
            }
            
        except Exception as e:
            return {'error': f'Ridge analysis failed: {str(e)}'}
    
    def analyze_feature_importance(self) -> Dict[str, float]:
        """Analyze individual feature importance using correlation analysis only"""
        print("📈 Analyzing feature importance...")
        
        if self.features_df is None:
            raise ValueError("Features not calculated yet. Run calculate_all_features() first.")
        
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['day', 'instrument', 'price', 'target']]
        
        # Prepare clean data
        clean_data = self.features_df.dropna(subset=['target'])
        X = clean_data[feature_cols].fillna(0).values
        y = clean_data['target'].values
        
        # Remove outliers
        y_p95 = np.percentile(y, 95)
        y_p5 = np.percentile(y, 5)
        mask = (y >= y_p5) & (y <= y_p95)
        X, y = X[mask], y[mask]
        
        print(f"✅ Using {len(X)} clean samples for feature analysis")
        
        importance_results = {}
        
        # Correlation-based importance
        print("  Calculating feature correlations...")
        for i, col in enumerate(feature_cols):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            feature_values = X[:, i]
            
            importance_results[col] = {
                'correlation': corr if not np.isnan(corr) else 0,
                'abs_correlation': abs(corr) if not np.isnan(corr) else 0,
                'mean': np.mean(feature_values),
                'std': np.std(feature_values),
                'min': np.min(feature_values),
                'max': np.max(feature_values),
                'median': np.median(feature_values),
                'percentile_25': np.percentile(feature_values, 25),
                'percentile_75': np.percentile(feature_values, 75),
                'non_zero_ratio': np.mean(feature_values != 0)
            }
        
        self.feature_importance = importance_results
        print("✅ Feature analysis complete")
        
        return importance_results
    
    def get_all_qlib_features(self) -> List[str]:
        """Get comprehensive list of all possible Qlib features"""
        print("📋 Generating comprehensive Qlib feature list...")
        
        # All possible Qlib features categorized
        qlib_features = []
        
        # Price-based features
        for period in [1, 2, 3, 5, 10, 15, 20, 30, 60, 120]:
            qlib_features.extend([
                f'Ref($close, {period})',
                f'Mean($close, {period})',
                f'Std($close, {period})',
                f'($close - Ref($close, {period})) / Ref($close, {period})',
                f'($close - Mean($close, {period})) / Std($close, {period})',
                f'Rank($close, {period})',
                f'Max($close, {period})',
                f'Min($close, {period})',
                f'($close - Min($close, {period})) / (Max($close, {period}) - Min($close, {period}))',
                f'Sum($close, {period})',
                f'Slope($close, {period})',
                f'Rsquare($close, {period})',
                f'Resi($close, {period})'
            ])
        
        # Volume-based features
        for period in [1, 2, 3, 5, 10, 15, 20, 30, 60]:
            qlib_features.extend([
                f'Ref($volume, {period})',
                f'Mean($volume, {period})',
                f'Std($volume, {period})',
                f'($volume - Ref($volume, {period})) / Ref($volume, {period})',
                f'Rank($volume, {period})',
                f'($volume - Mean($volume, {period})) / Std($volume, {period})',
                f'Corr($close, $volume, {period})',
                f'($volume * $close)',
                f'($volume * ($close - Ref($close, 1)))'
            ])
        
        # Technical indicators
        for period in [5, 10, 14, 20, 30]:
            qlib_features.extend([
                f'RSI($close, {period})',
                f'WILLR($close, {period})',
                f'CCI($close, $high, $low, {period})',
                f'CMF($close, $high, $low, $volume, {period})',
                f'PSY($close, {period})',
                f'BIAS($close, {period})',
                f'ROC($close, {period})',
                f'MOM($close, {period})',
                f'PPO($close, {period}, {period*2})',
                f'UOS($close, $high, $low, {period})',
                f'TRIX($close, {period})',
                f'DPO($close, {period})'
            ])
        
        # Moving averages and cross-overs
        for short in [5, 10, 20]:
            for long in [20, 30, 60, 120]:
                if short < long:
                    qlib_features.extend([
                        f'(Mean($close, {short}) - Mean($close, {long})) / Mean($close, {long})',
                        f'EMA($close, {short}) - EMA($close, {long})',
                        f'(EMA($close, {short}) > EMA($close, {long}))',
                        f'Mean(Mean($close, {short}) > Mean($close, {long}), 10)'
                    ])
        
        # Price position features
        for period in [10, 20, 30, 60]:
            qlib_features.extend([
                f'($close - Min($close, {period})) / (Max($close, {period}) - Min($close, {period}))',
                f'($close - Mean($close, {period})) / Std($close, {period})',
                f'Rank($close, {period}) / {period}',
                f'Quantile($close, {period}, 0.1)',
                f'Quantile($close, {period}, 0.9)'
            ])
        
        # Volatility features
        for period in [5, 10, 20, 30, 60]:
            qlib_features.extend([
                f'Std($close, {period})',
                f'Std($close, {period}) / Mean($close, {period})',
                f'Std(($close - Ref($close, 1)) / Ref($close, 1), {period})',
                f'Mean(Abs($close - Ref($close, 1)), {period})',
                f'Max($close, {period}) - Min($close, {period})',
                f'(Max($close, {period}) - Min($close, {period})) / Mean($close, {period})'
            ])
        
        # Cross-sectional features
        qlib_features.extend([
            'Rank($close)',
            'Rank($volume)', 
            'Rank(($close - Ref($close, 1)) / Ref($close, 1))',
            '($close - Mean($close)) / Std($close)',
            '($volume - Mean($volume)) / Std($volume)',
            'Rank(Mean($close, 20))',
            'Rank(Std($close, 20))',
            'Rank(RSI($close, 14))',
            'Rank(($close - Mean($close, 20)) / Std($close, 20))'
        ])
        
        print(f"✅ Generated {len(qlib_features)} comprehensive Qlib features")
        return qlib_features
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive markdown report - FEATURES ONLY"""
        print("📝 Generating comprehensive feature report...")
        
        if self.features_df is None or not self.feature_importance:
            raise ValueError("Features not calculated yet. Run calculate_all_features() and analyze_feature_importance() first.")
        
        # Sort features by absolute correlation
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1]['abs_correlation'],
            reverse=True
        )
        
        # Generate executive summary
        summary = self._generate_executive_summary(sorted_features)
        
        report = f"""# Comprehensive Qlib Feature & Ridge Analysis Report  
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

{summary}

## Dataset Summary
- **Data File**: {self.price_file}
- **Time Period**: {self.prices.shape[0]} days
- **Instruments**: {self.prices.shape[1]}
- **Total Features Calculated**: {len(self.feature_importance)}
- **Total Samples**: {len(self.features_df):,}

## All Features Analysis

### Complete Feature Statistics

| Rank | Feature Name | Correlation | Abs Corr | Mean | Std | Min | Max | Median | Non-Zero % |
|------|--------------|-------------|----------|------|-----|-----|-----|--------|------------|"""

        # Add all features
        for i, (feature_name, stats) in enumerate(sorted_features):
            corr = stats['correlation']
            abs_corr = stats['abs_correlation']
            mean_val = stats['mean']
            std_val = stats['std']
            min_val = stats['min']
            max_val = stats['max']
            median_val = stats['median']
            nonzero_pct = stats['non_zero_ratio'] * 100
            
            report += f"\n| {i+1:4d} | {feature_name[:30]:30s} | {corr:11.6f} | {abs_corr:8.6f} | {mean_val:6.3f} | {std_val:5.3f} | {min_val:5.2f} | {max_val:5.2f} | {median_val:8.3f} | {nonzero_pct:8.1f}% |"
        
        # Add individual stock analysis section  
        if self.individual_stock_features:
            report += f"""

## Individual Stock Analysis

### All 50 Stocks Feature Analysis

| Stock ID | Total Samples | Mean Return | Volatility | Ridge IC | Ridge Alpha | Top Corr Features | Top Ridge Features |
|----------|---------------|-------------|------------|----------|-------------|-------------------|-------------------|"""
            
            for stock_id, stats in self.individual_stock_features.items():
                # Create a summary of correlation-based features
                top_corr_str = "; ".join([f"{feat[0]}({feat[1]:.3f})" for feat in stats['ranked_features'][:5]])
                
                # Ridge analysis results
                ridge_results = stats.get('ridge_analysis', {})
                if 'error' in ridge_results:
                    ridge_ic = 0
                    ridge_alpha = 'ERROR'
                    top_ridge_str = 'Ridge analysis failed'
                else:
                    ridge_ic = ridge_results.get('information_coefficient', 0)
                    ridge_alpha = ridge_results.get('best_alpha', 'N/A')
                    if isinstance(ridge_alpha, float):
                        ridge_alpha = f"{ridge_alpha:.1f}"
                    
                    # Top Ridge features
                    ridge_features = ridge_results.get('ridge_ranked_features', [])
                    if ridge_features:
                        top_ridge_str = "; ".join([f"{feat[0]}({feat[1]:.3f})" for feat in ridge_features[:5]])
                    else:
                        top_ridge_str = 'No features'
                
                report += f"\n| {stock_id:8d} | {stats['total_samples']:13d} | {stats['mean_return']:11.6f} | {stats['return_volatility']:10.6f} | {ridge_ic:8.4f} | {ridge_alpha:11s} | {top_corr_str[:40]:40s} | {top_ridge_str[:40]:40s} |"
            
            # Most predictable stocks based on Ridge IC
            predictable_stocks = [(stock_id, stats) for stock_id, stats in self.individual_stock_features.items() 
                                if 'ridge_analysis' in stats and 'information_coefficient' in stats['ridge_analysis']]
            predictable_stocks.sort(key=lambda x: abs(x[1]['ridge_analysis']['information_coefficient']), reverse=True)
            
            report += f"""

### Top 10 Most Predictable Stocks (by Ridge IC)

| Rank | Stock ID | Ridge IC | Best Alpha | Top Ridge Feature | Ridge Coef |
|------|----------|----------|------------|-------------------|------------|"""
            
            for i, (stock_id, stats) in enumerate(predictable_stocks[:10]):
                ridge_results = stats['ridge_analysis']
                ridge_ic = ridge_results.get('information_coefficient', 0)
                best_alpha = ridge_results.get('best_alpha', 'N/A')
                if isinstance(best_alpha, float):
                    best_alpha = f"{best_alpha:.1f}"
                
                ridge_features = ridge_results.get('ridge_ranked_features', [])
                if ridge_features:
                    top_feature = ridge_features[0][0]
                    top_coef = ridge_features[0][1]
                else:
                    top_feature = 'N/A'
                    top_coef = 0
                
                report += f"\n| {i+1:4d} | {stock_id:8d} | {ridge_ic:8.4f} | {best_alpha:10s} | {top_feature[:30]:30s} | {top_coef:10.4f} |"

            # Complete feature breakdown for each stock
            report += f"""

### Complete Feature Analysis by Stock

"""
            for stock_id, stats in self.individual_stock_features.items():
                ridge_results = stats.get('ridge_analysis', {})
                ridge_ic = ridge_results.get('information_coefficient', 0)
                
                report += f"""
#### Stock {stock_id} - Ridge IC: {ridge_ic:.4f}

##### Correlation-based Feature Rankings
| Rank | Feature Name | Correlation | Mean | Std | Min | Max | Median |
|------|--------------|-------------|------|-----|-----|-----|--------|"""
                
                for i, (feature_name, corr) in enumerate(stats['ranked_features'][:20]):  # Top 20 only
                    feat_stats = stats['feature_statistics'][feature_name]
                    report += f"\n| {i+1:4d} | {feature_name[:35]:35s} | {corr:11.6f} | {feat_stats['mean']:6.3f} | {feat_stats['std']:5.3f} | {feat_stats['min']:5.2f} | {feat_stats['max']:5.2f} | {feat_stats['median']:8.3f} |"
                
                # Ridge-based rankings
                if 'ridge_ranked_features' in ridge_results and 'error' not in ridge_results:
                    report += f"""

##### Ridge Regression Feature Rankings  
| Rank | Feature Name | Ridge Coefficient | Correlation |
|------|--------------|-------------------|-------------|"""
                    
                    for i, (feature_name, coef) in enumerate(ridge_results['ridge_ranked_features'][:20]):  # Top 20 only
                        corr = stats['all_feature_correlations'].get(feature_name, 0)
                        report += f"\n| {i+1:4d} | {feature_name[:35]:35s} | {coef:17.6f} | {corr:11.6f} |"
                elif 'error' in ridge_results:
                    report += f"""

##### Ridge Regression Analysis
**Error**: {ridge_results['error']}"""
                
                report += "\n"

        report += f"""

## Feature Engineering Insights

### Top Performing Features Overall
"""
        for i, (feature_name, stats) in enumerate(sorted_features[:20]):
            report += f"{i+1}. **{feature_name}** (Correlation: {stats['abs_correlation']:.6f})\n"

        # Ridge analysis summary
        if self.individual_stock_features:
            ridge_ics = []
            for stock_id, stats in self.individual_stock_features.items():
                ridge_results = stats.get('ridge_analysis', {})
                if 'information_coefficient' in ridge_results:
                    ridge_ics.append(abs(ridge_results['information_coefficient']))
            
            if ridge_ics:
                avg_ridge_ic = np.mean(ridge_ics)
                max_ridge_ic = np.max(ridge_ics)
                min_ridge_ic = np.min(ridge_ics)
                
                report += f"""

### Ridge Regression Analysis Summary
- **Average Ridge IC across all stocks**: {avg_ridge_ic:.4f}
- **Best Ridge IC achieved**: {max_ridge_ic:.4f}
- **Worst Ridge IC**: {min_ridge_ic:.4f}
- **Number of stocks with IC > 0.02**: {sum(1 for ic in ridge_ics if ic > 0.02)}
- **Number of stocks with IC > 0.01**: {sum(1 for ic in ridge_ics if ic > 0.01)}
"""

        report += f"""

### Feature Categories Summary
- **Total Features Calculated**: {len(self.feature_importance)}
- **Price-based features**: Multiple timeframes (1d to 120d)
- **Volatility features**: Standard deviation and volatility ratios
- **Technical indicators**: RSI, Williams %R, Bollinger Bands, MACD, etc.
- **Cross-sectional features**: Rankings and market-relative metrics
- **Trend analysis**: Linear slopes and R-squared measures
- **Statistical features**: Skewness, kurtosis, percentile analysis

## Raw Data Export

### All Feature Correlations (CSV Format)
```csv
Feature_Name,Correlation,Abs_Correlation,Mean,Std,Min,Max,Median,Non_Zero_Ratio"""

        for feature_name, stats in sorted_features:
            report += f"\n{feature_name},{stats['correlation']},{stats['abs_correlation']},{stats['mean']},{stats['std']},{stats['min']},{stats['max']},{stats['median']},{stats['non_zero_ratio']}"

        report += f"""
```

## Implementation Notes
- Analyzed {len(self.features_df):,} total samples across {self.prices.shape[1]} instruments
- Applied outlier removal (5th-95th percentile)
- All features cleaned and normalized for missing/infinite values
- Individual stock analysis provides stock-specific feature rankings

---
*Comprehensive feature analysis completed using {len(self.feature_importance)} engineered features*
"""
        
        return report
    
    def _generate_executive_summary(self, sorted_features) -> str:
        """Generate executive summary of key findings"""
        
        # Top features
        top_3_features = [f"**{feat_name}** ({feat_stats['abs_correlation']:.4f})" for feat_name, feat_stats in sorted_features[:3]]
        
        # Ridge analysis summary
        ridge_summary = ""
        if self.individual_stock_features:
            ridge_ics = []
            predictable_stocks = []
            
            for stock_id, stats in self.individual_stock_features.items():
                ridge_results = stats.get('ridge_analysis', {})
                if 'information_coefficient' in ridge_results and 'error' not in ridge_results:
                    ic = ridge_results['information_coefficient']
                    ridge_ics.append(abs(ic))
                    if abs(ic) > 0.02:
                        predictable_stocks.append((stock_id, abs(ic)))
            
            if ridge_ics:
                avg_ic = np.mean(ridge_ics)
                max_ic = np.max(ridge_ics)
                strong_stocks = len([ic for ic in ridge_ics if ic > 0.02])
                moderate_stocks = len([ic for ic in ridge_ics if ic > 0.01])
                
                predictable_stocks.sort(key=lambda x: x[1], reverse=True)
                top_stocks = [f"Stock {stock_id} (IC: {ic:.4f})" for stock_id, ic in predictable_stocks[:3]]
                
                ridge_summary = f"""
### 🎯 Key Findings

**Ridge Regression Performance:**
- Average Information Coefficient: **{avg_ic:.4f}**
- Best IC achieved: **{max_ic:.4f}**
- {strong_stocks} stocks with strong predictability (IC > 0.02)
- {moderate_stocks} stocks with moderate predictability (IC > 0.01)

**Most Predictable Stocks:**
{', '.join(top_stocks) if top_stocks else 'None with IC > 0.02'}

**Top Performing Features:**
{', '.join(top_3_features)}

### 📊 Quick Stats
- **Total Features Analyzed**: {len(self.feature_importance)}
- **Total Stock-Days**: {len(self.features_df):,}
- **Stocks Analyzed**: {len(self.individual_stock_features)}
- **Feature Categories**: Price-based, Volatility, Technical Indicators, Cross-sectional, Trend Analysis

### 💡 Recommendation
{'🟢 Strong signals detected - proceed with feature selection and strategy development' if max_ic > 0.03 else '🟡 Moderate signals - consider feature engineering or ensemble approaches' if max_ic > 0.015 else '🔴 Weak signals - review data quality and feature engineering approach'}
"""
            else:
                ridge_summary = f"""
### 🎯 Key Findings

**Ridge Regression Performance:**
- Ridge analysis failed or no valid results
- Using correlation-based analysis only

**Top Performing Features:**
{', '.join(top_3_features)}

### 📊 Quick Stats
- **Total Features Analyzed**: {len(self.feature_importance)}
- **Total Stock-Days**: {len(self.features_df):,}
- **Stocks Analyzed**: {len(self.individual_stock_features)}
- **Feature Categories**: Price-based, Volatility, Technical Indicators, Cross-sectional, Trend Analysis

### 💡 Recommendation
🔴 Ridge analysis incomplete - review feature engineering and data quality
"""
        
        return ridge_summary
    
    def save_results(self, report: str):
        """Save analysis results"""
        # Save markdown report
        with open('qlib_feature_ridge_analysis_report.md', 'w') as f:
            f.write(report)
        
        # Save detailed results as pickle
        results = {
            'feature_importance': self.feature_importance,
            'individual_stock_features': self.individual_stock_features,
            'features_df': self.features_df,
            'data_shape': self.prices.shape,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open('qlib_feature_analysis_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print("✅ Results saved:")
        print("  📄 Report: qlib_feature_ridge_analysis_report.md")
        print("  💾 Data: qlib_feature_analysis_results.pkl")

def main():
    """Run comprehensive feature analysis"""
    print("🚀 COMPREHENSIVE QLIB FEATURE ANALYSIS")
    print("="*60)
    
    try:
        # Initialize analyzer
        analyzer = ComprehensiveFeatureAnalyzer()
        
        # Load data
        analyzer.load_data()
        
        # Calculate all features
        analyzer.calculate_all_features(lookback=90)  # Use 90 days for faster processing
        
        # Analyze feature importance
        analyzer.analyze_feature_importance()
        
        # Analyze individual stock features
        analyzer.analyze_individual_stock_features()
        
        # Generate report
        report = analyzer.generate_comprehensive_report()
        
        # Save results
        analyzer.save_results(report)
        
        print("🎉 Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
