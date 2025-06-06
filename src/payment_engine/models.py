"""
Payment Service Provider (PSP) Models and Configuration
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import random
import time

class PaymentMethod(Enum):
    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    BNPL = "bnpl"

class TransactionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    DECLINED = "declined"

@dataclass
class PSPConfig:
    """Configuration for Payment Service Provider"""
    name: str
    base_success_rate: float  # 0.0 to 1.0
    base_cost_percentage: float  # Cost as percentage of transaction
    avg_latency_ms: int
    supported_methods: List[PaymentMethod]
    supported_countries: List[str]
    daily_limit: int
    current_load: int = 0
    
    # Performance characteristics
    success_rate_by_method: Dict[PaymentMethod, float] = field(default_factory=dict)
    success_rate_by_country: Dict[str, float] = field(default_factory=dict)
    
    def get_success_rate(self, method: PaymentMethod, country: str, amount: float) -> float:
        """Calculate dynamic success rate based on various factors"""
        base_rate = self.base_success_rate
        
        # Adjust by payment method
        method_modifier = self.success_rate_by_method.get(method, 1.0)
        
        # Adjust by country
        country_modifier = self.success_rate_by_country.get(country, 1.0)
        
        # Adjust by current load
        load_factor = 1.0 - (self.current_load / self.daily_limit) * 0.1
        
        # Adjust by amount (higher amounts might have lower success rates)
        amount_factor = 1.0 - min(amount / 100000, 0.1)  # Reduce by up to 10% for high amounts
        
        final_rate = base_rate * method_modifier * country_modifier * load_factor * amount_factor
        return min(max(final_rate, 0.0), 1.0)
    
    def get_cost(self, amount: float) -> float:
        """Calculate transaction cost"""
        return amount * (self.base_cost_percentage / 100)
    
    def get_latency(self) -> int:
        """Get simulated latency with some randomness"""
        return self.avg_latency_ms + random.randint(-50, 100)

# Pre-configured PSPs mimicking real-world providers
DEFAULT_PSPS = [
    PSPConfig(
        name="Razorpay",
        base_success_rate=0.94,
        base_cost_percentage=2.0,
        avg_latency_ms=450,
        supported_methods=[PaymentMethod.CARD, PaymentMethod.UPI, PaymentMethod.NETBANKING, PaymentMethod.WALLET],
        supported_countries=["IN", "MY", "SG"],
        daily_limit=100000,
        success_rate_by_method={
            PaymentMethod.UPI: 0.98,
            PaymentMethod.CARD: 0.92,
            PaymentMethod.NETBANKING: 0.89,
            PaymentMethod.WALLET: 0.95
        },
        success_rate_by_country={
            "IN": 1.0,
            "MY": 0.85,
            "SG": 0.88
        }
    ),
    PSPConfig(
        name="Stripe",
        base_success_rate=0.96,
        base_cost_percentage=2.9,
        avg_latency_ms=320,
        supported_methods=[PaymentMethod.CARD],
        supported_countries=["US", "GB", "CA", "AU", "SG"],
        daily_limit=150000,
        success_rate_by_method={
            PaymentMethod.CARD: 0.96
        },
        success_rate_by_country={
            "US": 1.0,
            "GB": 0.94,
            "CA": 0.93,
            "AU": 0.91,
            "SG": 0.89
        }
    ),
    PSPConfig(
        name="PayU",
        base_success_rate=0.91,
        base_cost_percentage=1.8,
        avg_latency_ms=520,
        supported_methods=[PaymentMethod.CARD, PaymentMethod.UPI, PaymentMethod.NETBANKING, PaymentMethod.WALLET],
        supported_countries=["IN", "PL", "RO", "TR"],
        daily_limit=80000,
        success_rate_by_method={
            PaymentMethod.UPI: 0.93,
            PaymentMethod.CARD: 0.89,
            PaymentMethod.NETBANKING: 0.87,
            PaymentMethod.WALLET: 0.92
        },
        success_rate_by_country={
            "IN": 1.0,
            "PL": 0.82,
            "RO": 0.80,
            "TR": 0.78
        }
    ),
    PSPConfig(
        name="Cashfree",
        base_success_rate=0.93,
        base_cost_percentage=1.9,
        avg_latency_ms=380,
        supported_methods=[PaymentMethod.CARD, PaymentMethod.UPI, PaymentMethod.NETBANKING],
        supported_countries=["IN"],
        daily_limit=70000,
        success_rate_by_method={
            PaymentMethod.UPI: 0.96,
            PaymentMethod.CARD: 0.91,
            PaymentMethod.NETBANKING: 0.88
        },
        success_rate_by_country={
            "IN": 1.0
        }
    ),
    PSPConfig(
        name="Paytm",
        base_success_rate=0.89,
        base_cost_percentage=1.5,
        avg_latency_ms=600,
        supported_methods=[PaymentMethod.WALLET, PaymentMethod.UPI, PaymentMethod.CARD],
        supported_countries=["IN"],
        daily_limit=90000,
        success_rate_by_method={
            PaymentMethod.WALLET: 0.95,
            PaymentMethod.UPI: 0.91,
            PaymentMethod.CARD: 0.85
        },
        success_rate_by_country={
            "IN": 1.0
        }
    )
]

@dataclass
class Transaction:
    """Represents a payment transaction"""
    id: str
    amount: float
    currency: str
    payment_method: PaymentMethod
    country: str
    merchant_id: str
    timestamp: float = field(default_factory=time.time)
    
    # Routing information
    selected_psp: Optional[str] = None
    backup_psps: List[str] = field(default_factory=list)
    
    # Result information
    status: TransactionStatus = TransactionStatus.PENDING
    processing_time_ms: Optional[int] = None
    actual_cost: Optional[float] = None
    failure_reason: Optional[str] = None
    fraud_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert transaction to dictionary for serialization"""
        return {
            "id": self.id,
            "amount": self.amount,
            "currency": self.currency,
            "payment_method": self.payment_method.value,
            "country": self.country,
            "merchant_id": self.merchant_id,
            "timestamp": self.timestamp,
            "selected_psp": self.selected_psp,
            "backup_psps": self.backup_psps,
            "status": self.status.value,
            "processing_time_ms": self.processing_time_ms,
            "actual_cost": self.actual_cost,
            "failure_reason": self.failure_reason,
            "fraud_score": self.fraud_score
        }
