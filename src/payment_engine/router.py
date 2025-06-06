"""
Intelligent Payment Router with ML-based routing decisions
"""
import asyncio
import logging
import random
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from .models import Transaction, PSPConfig, PaymentMethod, TransactionStatus, DEFAULT_PSPS

logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    """Represents a routing decision with reasoning"""
    primary_psp: str
    backup_psps: List[str]
    expected_success_rate: float
    expected_cost: float
    expected_latency: int
    confidence_score: float
    reasoning: str

class PaymentRouter:
    """Intelligent Payment Router with ML-based decisions"""
    
    def __init__(self, psps: List[PSPConfig] = None):
        self.psps = {psp.name: psp for psp in (psps or DEFAULT_PSPS)}
        self.transaction_history = []
        self.routing_model = None
        self.feature_encoders = {}
        self.performance_metrics = {}
        
        # Initialize ML model
        self._initialize_ml_model()
        
        # Initialize performance tracking
        self._initialize_performance_metrics()
    
    def _initialize_ml_model(self):
        """Initialize ML model for routing decisions"""
        self.routing_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Initialize encoders
        self.feature_encoders = {
            'payment_method': LabelEncoder(),
            'country': LabelEncoder(),
            'merchant_id': LabelEncoder()
        }
        
        # Generate initial training data
        self._generate_synthetic_training_data()
    
    def _initialize_performance_metrics(self):
        """Initialize performance tracking for each PSP"""
        for psp_name in self.psps.keys():
            self.performance_metrics[psp_name] = {
                'total_transactions': 0,
                'successful_transactions': 0,
                'total_cost': 0.0,
                'total_latency': 0,
                'success_rate': 0.0,
                'avg_cost': 0.0,
                'avg_latency': 0.0,
                'last_updated': time.time()
            }
    
    def _generate_synthetic_training_data(self):
        """Generate synthetic training data for initial ML model"""
        training_data = []
        
        # Generate diverse transaction scenarios
        payment_methods = list(PaymentMethod)
        countries = ["IN", "US", "GB", "SG", "AU", "CA"]
        merchants = [f"merchant_{i}" for i in range(1, 21)]
        
        for _ in range(5000):
            # Generate random transaction
            method = random.choice(payment_methods)
            country = random.choice(countries)
            merchant = random.choice(merchants)
            amount = random.uniform(10, 10000)
            
            # Find compatible PSPs
            compatible_psps = [
                psp for psp in self.psps.values()
                if method in psp.supported_methods and country in psp.supported_countries
            ]
            
            if not compatible_psps:
                continue
            
            # Simulate best PSP selection (for training)
            best_psp = max(compatible_psps, key=lambda p: p.get_success_rate(method, country, amount))
            
            training_data.append({
                'payment_method': method.value,
                'country': country,
                'merchant_id': merchant,
                'amount': amount,
                'hour': random.randint(0, 23),
                'day_of_week': random.randint(0, 6),
                'best_psp': best_psp.name
            })
        
        # Train initial model
        if training_data:
            self._train_model(training_data)
    
    def _train_model(self, training_data: List[Dict]):
        """Train the ML model with transaction data"""
        df = pd.DataFrame(training_data)
        
        # Prepare features
        features = []
        for _, row in df.iterrows():
            feature_vector = self._extract_features(
                row['payment_method'],
                row['country'],
                row['merchant_id'],
                row['amount'],
                row['hour'],
                row['day_of_week']
            )
            features.append(feature_vector)
        
        features = np.array(features)
        targets = df['best_psp'].values
        
        # Train model
        self.routing_model.fit(features, targets)
        logger.info(f"Trained routing model with {len(training_data)} samples")
    
    def _extract_features(self, payment_method: str, country: str, merchant_id: str, 
                         amount: float, hour: int, day_of_week: int) -> np.ndarray:
        """Extract features for ML model"""
        # Encode categorical variables
        try:
            method_encoded = self.feature_encoders['payment_method'].transform([payment_method])[0]
        except ValueError:
            # Handle unseen values
            method_encoded = -1
        
        try:
            country_encoded = self.feature_encoders['country'].transform([country])[0]
        except ValueError:
            country_encoded = -1
        
        try:
            merchant_encoded = self.feature_encoders['merchant_id'].transform([merchant_id])[0]
        except ValueError:
            merchant_encoded = -1
        
        # Create feature vector
        features = np.array([
            method_encoded,
            country_encoded,
            merchant_encoded,
            np.log1p(amount),  # Log transform amount
            hour,
            day_of_week,
            amount / 1000,  # Normalized amount
        ])
        
        return features
    
    async def route_transaction(self, transaction: Transaction) -> RoutingDecision:
        """Route transaction to optimal PSP"""
        # Find compatible PSPs
        compatible_psps = self._find_compatible_psps(transaction)
        
        if not compatible_psps:
            return RoutingDecision(
                primary_psp="",
                backup_psps=[],
                expected_success_rate=0.0,
                expected_cost=0.0,
                expected_latency=0,
                confidence_score=0.0,
                reasoning="No compatible PSPs found"
            )
        
        # Score each compatible PSP
        psp_scores = []
        for psp in compatible_psps:
            score = await self._score_psp(psp, transaction)
            psp_scores.append((psp, score))
        
        # Sort by score (higher is better)
        psp_scores.sort(key=lambda x: x[1]['composite_score'], reverse=True)
        
        # Select primary and backup PSPs
        primary_psp = psp_scores[0][0]
        backup_psps = [psp for psp, _ in psp_scores[1:3]]  # Top 2 backups
        
        # Calculate expected metrics
        primary_score = psp_scores[0][1]
        
        return RoutingDecision(
            primary_psp=primary_psp.name,
            backup_psps=[psp.name for psp in backup_psps],
            expected_success_rate=primary_score['success_rate'],
            expected_cost=primary_score['cost'],
            expected_latency=primary_score['latency'],
            confidence_score=primary_score['confidence'],
            reasoning=primary_score['reasoning']
        )
    
    def _find_compatible_psps(self, transaction: Transaction) -> List[PSPConfig]:
        """Find PSPs that can handle this transaction"""
        compatible = []
        
        for psp in self.psps.values():
            # Check payment method support
            if transaction.payment_method not in psp.supported_methods:
                continue
            
            # Check country support
            if transaction.country not in psp.supported_countries:
                continue
            
            # Check daily limit
            if psp.current_load >= psp.daily_limit:
                continue
            
            compatible.append(psp)
        
        return compatible
    
    async def _score_psp(self, psp: PSPConfig, transaction: Transaction) -> Dict:
        """Score a PSP for a given transaction"""
        # Get base metrics
        success_rate = psp.get_success_rate(
            transaction.payment_method,
            transaction.country,
            transaction.amount
        )
        cost = psp.get_cost(transaction.amount)
        latency = psp.get_latency()
        
        # Apply ML model prediction if available
        if self.routing_model is not None:
            try:
                current_time = time.time()
                hour = int((current_time % 86400) / 3600)
                day_of_week = int((current_time / 86400) % 7)
                
                features = self._extract_features(
                    transaction.payment_method.value,
                    transaction.country,
                    transaction.merchant_id,
                    transaction.amount,
                    hour,
                    day_of_week
                )
                
                # Get prediction probability
                prediction_proba = self.routing_model.predict_proba([features])
                psp_classes = self.routing_model.classes_
                
                if psp.name in psp_classes:
                    psp_index = list(psp_classes).index(psp.name)
                    ml_confidence = prediction_proba[0][psp_index]
                else:
                    ml_confidence = 0.5  # Default confidence
                
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                ml_confidence = 0.5
        else:
            ml_confidence = 0.5
        
        # Apply historical performance
        historical_metrics = self.performance_metrics.get(psp.name, {})
        historical_success_rate = historical_metrics.get('success_rate', success_rate)
        
        # Weighted success rate (70% current, 30% historical)
        adjusted_success_rate = 0.7 * success_rate + 0.3 * historical_success_rate
        
        # Calculate composite score
        # Weights: success_rate (50%), cost (20%), latency (15%), ML confidence (15%)
        normalized_cost = 1.0 - min(cost / (transaction.amount * 0.05), 1.0)  # Invert cost
        normalized_latency = 1.0 - min(latency / 1000, 1.0)  # Invert latency
        
        composite_score = (
            adjusted_success_rate * 0.5 +
            normalized_cost * 0.2 +
            normalized_latency * 0.15 +
            ml_confidence * 0.15
        )
        
        reasoning = f"Success: {adjusted_success_rate:.1%}, Cost: ${cost:.2f}, Latency: {latency}ms, ML: {ml_confidence:.1%}"
        
        return {
            'success_rate': adjusted_success_rate,
            'cost': cost,
            'latency': latency,
            'ml_confidence': ml_confidence,
            'composite_score': composite_score,
            'confidence': ml_confidence,
            'reasoning': reasoning
        }
    
    async def process_transaction(self, transaction: Transaction) -> Transaction:
        """Process a transaction through the routing engine"""
        # Route transaction
        routing_decision = await self.route_transaction(transaction)
        
        if not routing_decision.primary_psp:
            transaction.status = TransactionStatus.FAILED
            transaction.failure_reason = "No compatible PSP found"
            return transaction
        
        # Update transaction with routing decision
        transaction.selected_psp = routing_decision.primary_psp
        transaction.backup_psps = routing_decision.backup_psps
        
        # Simulate transaction processing
        psp = self.psps[routing_decision.primary_psp]
        
        # Simulate processing time
        start_time = time.time()
        await asyncio.sleep(psp.get_latency() / 1000)  # Convert ms to seconds
        processing_time = int((time.time() - start_time) * 1000)
        
        # Determine success/failure
        success_rate = psp.get_success_rate(
            transaction.payment_method,
            transaction.country,
            transaction.amount
        )
        
        if random.random() < success_rate:
            transaction.status = TransactionStatus.SUCCESS
            transaction.actual_cost = psp.get_cost(transaction.amount)
            psp.current_load += 1
        else:
            transaction.status = TransactionStatus.FAILED
            transaction.failure_reason = "PSP declined transaction"
        
        transaction.processing_time_ms = processing_time
        
        # Update performance metrics
        self._update_performance_metrics(transaction)
        
        # Add to history for ML training
        self.transaction_history.append(transaction)
        
        return transaction
    
    def _update_performance_metrics(self, transaction: Transaction):
        """Update PSP performance metrics"""
        psp_name = transaction.selected_psp
        if not psp_name or psp_name not in self.performance_metrics:
            return
        
        metrics = self.performance_metrics[psp_name]
        metrics['total_transactions'] += 1
        
        if transaction.status == TransactionStatus.SUCCESS:
            metrics['successful_transactions'] += 1
            if transaction.actual_cost:
                metrics['total_cost'] += transaction.actual_cost
        
        if transaction.processing_time_ms:
            metrics['total_latency'] += transaction.processing_time_ms
        
        # Recalculate averages
        if metrics['total_transactions'] > 0:
            metrics['success_rate'] = metrics['successful_transactions'] / metrics['total_transactions']
            metrics['avg_cost'] = metrics['total_cost'] / max(metrics['successful_transactions'], 1)
            metrics['avg_latency'] = metrics['total_latency'] / metrics['total_transactions']
        
        metrics['last_updated'] = time.time()
    
    def get_analytics_summary(self) -> Dict:
        """Get comprehensive analytics summary"""
        total_transactions = len(self.transaction_history)
        successful_transactions = sum(1 for t in self.transaction_history if t.status == TransactionStatus.SUCCESS)
        
        return {
            'total_transactions': total_transactions,
            'successful_transactions': successful_transactions,
            'overall_success_rate': successful_transactions / max(total_transactions, 1),
            'psp_performance': self.performance_metrics,
            'recent_transactions': [t.to_dict() for t in self.transaction_history[-10:]],
            'timestamp': time.time()
        }
