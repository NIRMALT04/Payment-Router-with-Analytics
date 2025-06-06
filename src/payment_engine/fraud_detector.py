"""
Real-time Fraud Detection Module
"""
import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

from .models import Transaction, PaymentMethod

@dataclass
class FraudAlert:
    """Represents a fraud detection alert"""
    transaction_id: str
    fraud_score: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_factors: List[str]
    recommended_action: str
    timestamp: float

class FraudDetector:
    """Real-time fraud detection system"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        self.scaler = StandardScaler()
        self.transaction_history = []
        self.fraud_patterns = self._initialize_fraud_patterns()
        self.velocity_tracker = {}  # Track transaction velocity per merchant/user
        self.is_trained = False
        
        # Initialize with some baseline data
        self._initialize_baseline_model()
    
    def _initialize_fraud_patterns(self) -> Dict:
        """Initialize known fraud patterns and rules"""
        return {
            "high_amount_thresholds": {
                PaymentMethod.CARD: 10000,
                PaymentMethod.UPI: 5000,
                PaymentMethod.NETBANKING: 15000,
                PaymentMethod.WALLET: 3000,
                PaymentMethod.BNPL: 20000
            },
            "suspicious_countries": ["NG", "BD", "PK", "UA", "RU"],
            "round_amount_patterns": [1000, 2000, 5000, 10000, 25000, 50000],
            "velocity_limits": {
                "transactions_per_minute": 10,
                "amount_per_hour": 50000
            },
            "time_patterns": {
                "suspicious_hours": [2, 3, 4, 5],  # 2-5 AM
                "weekend_multiplier": 1.5
            }
        }
    
    def _initialize_baseline_model(self):
        """Initialize ML model with baseline data"""
        # Generate baseline "normal" transactions for training
        baseline_data = []
        
        for _ in range(1000):
            # Create normal transaction features
            features = [
                random.uniform(10, 1000),  # amount
                random.randint(6, 22),     # hour (business hours)
                random.randint(0, 4),      # weekday
                random.uniform(0, 100),    # merchant_risk_score
                random.uniform(0, 50),     # velocity_score
                random.randint(1, 5),      # payment_method_encoded
                random.uniform(0.1, 2.0),  # amount_velocity_ratio
            ]
            baseline_data.append(features)
        
        # Add some anomalies
        for _ in range(100):
            features = [
                random.uniform(5000, 50000),  # High amount
                random.randint(0, 6),         # Unusual hours
                random.randint(5, 6),         # Weekend
                random.uniform(50, 100),      # High merchant risk
                random.uniform(100, 500),     # High velocity
                random.randint(1, 5),         # payment_method_encoded
                random.uniform(5.0, 20.0),    # High velocity ratio
            ]
            baseline_data.append(features)
        
        # Train the model
        baseline_array = np.array(baseline_data)
        self.scaler.fit(baseline_array)
        scaled_data = self.scaler.transform(baseline_array)
        self.isolation_forest.fit(scaled_data)
        self.is_trained = True
    
    def extract_features(self, transaction: Transaction) -> np.ndarray:
        """Extract fraud detection features from transaction"""
        # Time-based features
        current_time = transaction.timestamp
        hour = int((current_time % 86400) / 3600)
        day_of_week = int((current_time / 86400) % 7)
        
        # Amount features
        amount = transaction.amount
        log_amount = np.log1p(amount)
        
        # Merchant risk score (based on historical data)
        merchant_risk_score = self._calculate_merchant_risk(transaction.merchant_id)
        
        # Velocity features
        velocity_score = self._calculate_velocity_score(transaction)
        
        # Payment method encoding
        payment_method_encoded = {
            PaymentMethod.CARD: 1,
            PaymentMethod.UPI: 2,
            PaymentMethod.NETBANKING: 3,
            PaymentMethod.WALLET: 4,
            PaymentMethod.BNPL: 5
        }.get(transaction.payment_method, 0)
        
        # Country risk (binary)
        country_risk = 1 if transaction.country in self.fraud_patterns["suspicious_countries"] else 0
        
        # Round amount indicator
        is_round_amount = 1 if amount in self.fraud_patterns["round_amount_patterns"] else 0
        
        # Time risk
        is_suspicious_hour = 1 if hour in self.fraud_patterns["time_patterns"]["suspicious_hours"] else 0
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Amount velocity ratio
        recent_velocity = self._get_recent_velocity(transaction.merchant_id)
        amount_velocity_ratio = amount / max(recent_velocity, 1)
        
        features = np.array([
            log_amount,
            hour,
            day_of_week,
            merchant_risk_score,
            velocity_score,
            payment_method_encoded,
            country_risk,
            is_round_amount,
            is_suspicious_hour,
            is_weekend,
            amount_velocity_ratio
        ])
        
        return features
    
    def _calculate_merchant_risk(self, merchant_id: str) -> float:
        """Calculate risk score for merchant based on history"""
        merchant_transactions = [
            t for t in self.transaction_history
            if t.merchant_id == merchant_id
        ]
        
        if not merchant_transactions:
            return 50.0  # Default medium risk for new merchants
        
        # Calculate metrics
        total_transactions = len(merchant_transactions)
        high_amount_count = sum(1 for t in merchant_transactions if t.amount > 5000)
        
        # Recent fraud incidents (simulated)
        recent_fraud_rate = random.uniform(0, 0.1) if total_transactions > 10 else 0.05
        
        # Risk factors
        high_amount_ratio = high_amount_count / total_transactions
        volume_risk = min(sum(t.amount for t in merchant_transactions) / 100000, 1.0)
        
        risk_score = (
            recent_fraud_rate * 40 +
            high_amount_ratio * 30 +
            volume_risk * 30
        )
        
        return min(risk_score, 100.0)
    
    def _calculate_velocity_score(self, transaction: Transaction) -> float:
        """Calculate transaction velocity score"""
        current_time = transaction.timestamp
        merchant_id = transaction.merchant_id
        
        # Get recent transactions for this merchant
        recent_transactions = [
            t for t in self.transaction_history
            if t.merchant_id == merchant_id and current_time - t.timestamp < 3600  # Last hour
        ]
        
        if not recent_transactions:
            return 0.0
        
        # Calculate velocity metrics
        transaction_count = len(recent_transactions)
        total_amount = sum(t.amount for t in recent_transactions)
        
        # Score based on limits
        count_score = min(transaction_count / self.fraud_patterns["velocity_limits"]["transactions_per_minute"], 1.0) * 50
        amount_score = min(total_amount / self.fraud_patterns["velocity_limits"]["amount_per_hour"], 1.0) * 50
        
        return count_score + amount_score
    
    def _get_recent_velocity(self, merchant_id: str) -> float:
        """Get recent transaction velocity for merchant"""
        current_time = time.time()
        recent_transactions = [
            t for t in self.transaction_history
            if t.merchant_id == merchant_id and current_time - t.timestamp < 1800  # Last 30 minutes
        ]
        
        if not recent_transactions:
            return 1.0
        
        return sum(t.amount for t in recent_transactions) / len(recent_transactions)
    
    async def analyze_transaction(self, transaction: Transaction) -> FraudAlert:
        """Analyze transaction for fraud indicators"""
        # Extract features
        features = self.extract_features(transaction)
        
        # Rule-based analysis
        rule_score, rule_factors = self._rule_based_analysis(transaction)
        
        # ML-based analysis
        ml_score = 0.5  # Default
        if self.is_trained:
            try:
                scaled_features = self.scaler.transform([features])
                anomaly_score = self.isolation_forest.decision_function(scaled_features)[0]
                # Convert to 0-1 scale (higher = more suspicious)
                ml_score = max(0, min(1, (0.5 - anomaly_score) * 2))
            except Exception as e:
                print(f"ML analysis failed: {e}")
        
        # Combine scores
        final_score = (rule_score * 0.6 + ml_score * 0.4)
        
        # Determine risk level and action
        risk_level, action = self._determine_risk_level(final_score)
        
        # Create fraud alert
        alert = FraudAlert(
            transaction_id=transaction.id,
            fraud_score=final_score,
            risk_level=risk_level,
            risk_factors=rule_factors,
            recommended_action=action,
            timestamp=time.time()
        )
        
        # Store fraud score in transaction
        transaction.fraud_score = final_score
        
        # Add to history
        self.transaction_history.append(transaction)
        
        # Keep only recent history (last 24 hours)
        cutoff_time = time.time() - 86400
        self.transaction_history = [
            t for t in self.transaction_history
            if t.timestamp > cutoff_time
        ]
        
        return alert
    
    def _rule_based_analysis(self, transaction: Transaction) -> Tuple[float, List[str]]:
        """Perform rule-based fraud analysis"""
        score = 0.0
        risk_factors = []
        
        # High amount check
        threshold = self.fraud_patterns["high_amount_thresholds"].get(transaction.payment_method, 10000)
        if transaction.amount > threshold:
            score += 0.3
            risk_factors.append(f"High amount: ${transaction.amount:.2f}")
        
        # Suspicious country
        if transaction.country in self.fraud_patterns["suspicious_countries"]:
            score += 0.4
            risk_factors.append(f"Suspicious country: {transaction.country}")
        
        # Round amount pattern
        if transaction.amount in self.fraud_patterns["round_amount_patterns"]:
            score += 0.2
            risk_factors.append(f"Round amount pattern: ${transaction.amount:.0f}")
        
        # Time-based checks
        hour = int((transaction.timestamp % 86400) / 3600)
        if hour in self.fraud_patterns["time_patterns"]["suspicious_hours"]:
            score += 0.25
            risk_factors.append(f"Suspicious hour: {hour}:00")
        
        # Velocity check
        velocity_score = self._calculate_velocity_score(transaction)
        if velocity_score > 50:
            score += 0.3
            risk_factors.append(f"High velocity: {velocity_score:.1f}")
        
        # Merchant risk
        merchant_risk = self._calculate_merchant_risk(transaction.merchant_id)
        if merchant_risk > 70:
            score += 0.2
            risk_factors.append(f"High-risk merchant: {merchant_risk:.1f}")
        
        return min(score, 1.0), risk_factors
    
    def _determine_risk_level(self, score: float) -> Tuple[str, str]:
        """Determine risk level and recommended action"""
        if score >= 0.8:
            return "CRITICAL", "BLOCK_TRANSACTION"
        elif score >= 0.6:
            return "HIGH", "MANUAL_REVIEW"
        elif score >= 0.4:
            return "MEDIUM", "ADDITIONAL_VERIFICATION"
        else:
            return "LOW", "ALLOW"
    
    def get_fraud_statistics(self) -> Dict:
        """Get fraud detection statistics"""
        if not self.transaction_history:
            return {"message": "No transaction history available"}
        
        # Calculate fraud rates by different dimensions
        total_transactions = len(self.transaction_history)
        high_risk_count = sum(1 for t in self.transaction_history if t.fraud_score and t.fraud_score >= 0.6)
        
        # Fraud by payment method
        fraud_by_method = {}
        for method in PaymentMethod:
            method_transactions = [t for t in self.transaction_history if t.payment_method == method]
            if method_transactions:
                high_risk = sum(1 for t in method_transactions if t.fraud_score and t.fraud_score >= 0.6)
                fraud_by_method[method.value] = {
                    "total": len(method_transactions),
                    "high_risk": high_risk,
                    "fraud_rate": high_risk / len(method_transactions)
                }
        
        # Fraud by country
        fraud_by_country = {}
        countries = set(t.country for t in self.transaction_history)
        for country in countries:
            country_transactions = [t for t in self.transaction_history if t.country == country]
            high_risk = sum(1 for t in country_transactions if t.fraud_score and t.fraud_score >= 0.6)
            fraud_by_country[country] = {
                "total": len(country_transactions),
                "high_risk": high_risk,
                "fraud_rate": high_risk / len(country_transactions)
            }
        
        return {
            "total_transactions": total_transactions,
            "high_risk_transactions": high_risk_count,
            "overall_fraud_rate": high_risk_count / total_transactions,
            "fraud_by_payment_method": fraud_by_method,
            "fraud_by_country": fraud_by_country,
            "model_status": "trained" if self.is_trained else "not_trained"
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent high-risk transactions"""
        high_risk_transactions = [
            t for t in self.transaction_history
            if t.fraud_score and t.fraud_score >= 0.6
        ]
        
        # Sort by fraud score descending
        high_risk_transactions.sort(key=lambda t: t.fraud_score, reverse=True)
        
        alerts = []
        for t in high_risk_transactions[:limit]:
            alerts.append({
                "transaction_id": t.id,
                "amount": t.amount,
                "currency": t.currency,
                "payment_method": t.payment_method.value,
                "country": t.country,
                "merchant_id": t.merchant_id,
                "fraud_score": t.fraud_score,
                "timestamp": t.timestamp
            })
        
        return alerts
