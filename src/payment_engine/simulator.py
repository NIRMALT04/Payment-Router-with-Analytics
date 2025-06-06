"""
Transaction Simulator for generating realistic payment scenarios
"""
import asyncio
import random
import time
import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from .models import Transaction, PaymentMethod

@dataclass
class TransactionPattern:
    """Defines a transaction generation pattern"""
    transactions_per_minute: int
    payment_method_distribution: Dict[PaymentMethod, float]
    country_distribution: Dict[str, float]
    amount_range: tuple
    merchant_pool: List[str]
    currency: str = "USD"

class TransactionSimulator:
    """Simulates realistic transaction patterns"""
    
    def __init__(self):
        self.is_running = False
        self.generated_transactions = []
        self.patterns = self._create_default_patterns()
        
    def _create_default_patterns(self) -> Dict[str, TransactionPattern]:
        """Create realistic transaction patterns for different scenarios"""
        return {
            "ecommerce_peak": TransactionPattern(
                transactions_per_minute=50,
                payment_method_distribution={
                    PaymentMethod.CARD: 0.6,
                    PaymentMethod.UPI: 0.25,
                    PaymentMethod.WALLET: 0.10,
                    PaymentMethod.NETBANKING: 0.05
                },
                country_distribution={
                    "IN": 0.7,
                    "US": 0.15,
                    "SG": 0.08,
                    "GB": 0.07
                },
                amount_range=(10, 5000),
                merchant_pool=[
                    "flipkart", "amazon_in", "myntra", "swiggy", "zomato",
                    "bigbasket", "nykaa", "paytm_mall", "shopclues", "snapdeal"
                ],
                currency="INR"
            ),
            
            "global_saas": TransactionPattern(
                transactions_per_minute=20,
                payment_method_distribution={
                    PaymentMethod.CARD: 0.85,
                    PaymentMethod.WALLET: 0.15
                },
                country_distribution={
                    "US": 0.4,
                    "GB": 0.2,
                    "CA": 0.15,
                    "AU": 0.1,
                    "SG": 0.15
                },
                amount_range=(29, 999),
                merchant_pool=[
                    "netflix", "spotify", "zoom", "slack", "notion",
                    "figma", "canva", "dropbox", "adobe", "microsoft"
                ],
                currency="USD"
            ),
            
            "gaming_microtransactions": TransactionPattern(
                transactions_per_minute=80,
                payment_method_distribution={
                    PaymentMethod.CARD: 0.4,
                    PaymentMethod.UPI: 0.3,
                    PaymentMethod.WALLET: 0.25,
                    PaymentMethod.BNPL: 0.05
                },
                country_distribution={
                    "IN": 0.5,
                    "US": 0.2,
                    "GB": 0.1,
                    "SG": 0.1,
                    "AU": 0.1
                },
                amount_range=(1, 100),
                merchant_pool=[
                    "pubg_mobile", "free_fire", "cod_mobile", "clash_royale",
                    "roblox", "fortnite", "minecraft", "among_us", "genshin"
                ],
                currency="USD"
            ),
            
            "food_delivery": TransactionPattern(
                transactions_per_minute=120,
                payment_method_distribution={
                    PaymentMethod.UPI: 0.4,
                    PaymentMethod.CARD: 0.3,
                    PaymentMethod.WALLET: 0.25,
                    PaymentMethod.BNPL: 0.05
                },
                country_distribution={
                    "IN": 0.8,
                    "SG": 0.1,
                    "US": 0.1
                },
                amount_range=(50, 1500),
                merchant_pool=[
                    "zomato", "swiggy", "uber_eats", "foodpanda", "dunzo",
                    "dominos", "mcdonalds", "kfc", "pizza_hut", "subway"
                ],
                currency="INR"
            )
        }
    
    def _weighted_choice(self, choices: Dict, total_weight: Optional[float] = None) -> any:
        """Make a weighted random choice"""
        if total_weight is None:
            total_weight = sum(choices.values())
        
        r = random.uniform(0, total_weight)
        upto = 0
        for choice, weight in choices.items():
            if upto + weight >= r:
                return choice
            upto += weight
        return list(choices.keys())[-1]  # Fallback
    
    def generate_transaction(self, pattern: TransactionPattern) -> Transaction:
        """Generate a single transaction based on pattern"""
        # Select payment method
        payment_method = self._weighted_choice(pattern.payment_method_distribution)
        
        # Select country
        country = self._weighted_choice(pattern.country_distribution)
        
        # Generate amount (with realistic distribution)
        min_amount, max_amount = pattern.amount_range
        
        # Use log-normal distribution for more realistic amounts
        mu = np.log(min_amount + (max_amount - min_amount) * 0.3)
        sigma = 0.8
        amount = np.random.lognormal(mu, sigma)
        amount = max(min_amount, min(amount, max_amount))
        amount = round(amount, 2)
        
        # Select merchant
        merchant = random.choice(pattern.merchant_pool)
        
        # Create transaction
        transaction = Transaction(
            id=str(uuid.uuid4()),
            amount=amount,
            currency=pattern.currency,
            payment_method=payment_method,
            country=country,
            merchant_id=merchant,
            timestamp=time.time()
        )
        
        return transaction
    
    def generate_fraud_transaction(self, pattern: TransactionPattern) -> Transaction:
        """Generate a potentially fraudulent transaction"""
        transaction = self.generate_transaction(pattern)
        
        # Make it suspicious
        fraud_indicators = [
            # High amount
            lambda t: setattr(t, 'amount', random.uniform(5000, 50000)),
            
            # Unusual country for merchant
            lambda t: setattr(t, 'country', random.choice(["NG", "BD", "PK", "UA"])),
            
            # Round amounts (often fraudulent)
            lambda t: setattr(t, 'amount', float(random.choice([1000, 2000, 5000, 10000]))),
            
            # Late night transactions
            lambda t: None  # We'll handle this differently
        ]
        
        # Apply 1-2 fraud indicators
        num_indicators = random.randint(1, 2)
        selected_indicators = random.sample(fraud_indicators, num_indicators)
        
        for indicator in selected_indicators:
            indicator(transaction)
        
        return transaction
    
    async def simulate_pattern(self, pattern_name: str, duration_minutes: int = 5) -> List[Transaction]:
        """Simulate transactions for a specific pattern"""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        pattern = self.patterns[pattern_name]
        transactions = []
        
        total_transactions = pattern.transactions_per_minute * duration_minutes
        interval = 60.0 / pattern.transactions_per_minute  # Seconds between transactions
        
        print(f"🚀 Starting simulation: {pattern_name}")
        print(f"📊 Generating {total_transactions} transactions over {duration_minutes} minutes")
        
        for i in range(total_transactions):
            # Add some variance to timing
            actual_interval = interval + random.uniform(-interval*0.3, interval*0.3)
            
            # Generate transaction
            if random.random() < 0.02:  # 2% fraud rate
                transaction = self.generate_fraud_transaction(pattern)
            else:
                transaction = self.generate_transaction(pattern)
            
            transactions.append(transaction)
            self.generated_transactions.append(transaction)
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"📈 Generated {i + 1}/{total_transactions} transactions")
            
            # Don't actually wait in simulation mode
            if duration_minutes > 10:  # Only add delays for long simulations
                await asyncio.sleep(min(actual_interval, 0.1))
        
        print(f"✅ Simulation complete: {len(transactions)} transactions generated")
        return transactions
    
    async def simulate_mixed_load(self, duration_minutes: int = 10) -> List[Transaction]:
        """Simulate mixed transaction patterns simultaneously"""
        print("🌐 Starting mixed load simulation...")
        
        # Create tasks for different patterns
        tasks = []
        
        # Peak ecommerce during business hours
        tasks.append(self.simulate_pattern("ecommerce_peak", duration_minutes // 2))
        
        # Continuous SaaS subscriptions
        tasks.append(self.simulate_pattern("global_saas", duration_minutes))
        
        # Gaming microtransactions (evening peak)
        tasks.append(self.simulate_pattern("gaming_microtransactions", duration_minutes // 3))
        
        # Food delivery rush
        tasks.append(self.simulate_pattern("food_delivery", duration_minutes // 4))
        
        # Run all patterns concurrently
        results = await asyncio.gather(*tasks)
        
        # Combine all transactions
        all_transactions = []
        for transaction_list in results:
            all_transactions.extend(transaction_list)
        
        # Sort by timestamp
        all_transactions.sort(key=lambda t: t.timestamp)
        
        print(f"🎯 Mixed simulation complete: {len(all_transactions)} total transactions")
        return all_transactions
    
    def get_pattern_analytics(self, pattern_name: str) -> Dict:
        """Get analytics for a specific pattern"""
        if pattern_name not in self.patterns:
            return {}
        
        pattern_transactions = [
            t for t in self.generated_transactions
            if t.merchant_id in self.patterns[pattern_name].merchant_pool
        ]
        
        if not pattern_transactions:
            return {"message": "No transactions found for this pattern"}
        
        # Calculate analytics
        total_volume = sum(t.amount for t in pattern_transactions)
        avg_amount = total_volume / len(pattern_transactions)
        
        payment_method_counts = {}
        country_counts = {}
        
        for t in pattern_transactions:
            method = t.payment_method.value
            payment_method_counts[method] = payment_method_counts.get(method, 0) + 1
            country_counts[t.country] = country_counts.get(t.country, 0) + 1
        
        return {
            "pattern_name": pattern_name,
            "total_transactions": len(pattern_transactions),
            "total_volume": round(total_volume, 2),
            "average_amount": round(avg_amount, 2),
            "payment_methods": payment_method_counts,
            "countries": country_counts,
            "currency": self.patterns[pattern_name].currency
        }
    
    def get_real_time_stats(self) -> Dict:
        """Get real-time transaction statistics"""
        recent_transactions = [
            t for t in self.generated_transactions
            if time.time() - t.timestamp < 300  # Last 5 minutes
        ]
        
        if not recent_transactions:
            return {
                "transactions_per_minute": 0,
                "total_volume": 0,
                "avg_amount": 0,
                "active_patterns": []
            }
        
        # Calculate TPS and volume
        time_span = max(time.time() - min(t.timestamp for t in recent_transactions), 1)
        tps = len(recent_transactions) / time_span * 60  # Transactions per minute
        
        total_volume = sum(t.amount for t in recent_transactions)
        avg_amount = total_volume / len(recent_transactions)
        
        # Identify active patterns
        active_merchants = set(t.merchant_id for t in recent_transactions)
        active_patterns = []
        
        for pattern_name, pattern in self.patterns.items():
            if any(merchant in active_merchants for merchant in pattern.merchant_pool):
                active_patterns.append(pattern_name)
        
        return {
            "transactions_per_minute": round(tps, 1),
            "total_volume": round(total_volume, 2),
            "avg_amount": round(avg_amount, 2),
            "active_patterns": active_patterns,
            "recent_transaction_count": len(recent_transactions)
        }
    
    def clear_history(self):
        """Clear transaction history"""
        self.generated_transactions.clear()
        print("🗑️ Transaction history cleared")
