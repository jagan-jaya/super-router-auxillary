"""
Core data models for the routing optimization system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
import uuid


class PaymentMethod(str, Enum):
    """Payment method types."""
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    WALLET = "wallet"
    UPI = "upi"
    BNPL = "bnpl"


class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    INR = "INR"
    JPY = "JPY"
    AUD = "AUD"
    CAD = "CAD"


class TransactionStatus(str, Enum):
    """Transaction status types."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    PROCESSING = "processing"
    DECLINED = "declined"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"
    CHARGEBACK = "chargeback"
    DISPUTED = "disputed"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    VOIDED = "voided"
    EXPIRED = "expired"


class ConnectorType(str, Enum):
    """Connector types."""
    GLOBAL = "global"
    LOCAL = "local"
    REGIONAL = "regional"


class RoutingAlgorithm(str, Enum):
    """Available routing algorithms."""
    SUCCESS_RATE = "SR"
    VOLUME_DISTRIBUTION = "V"
    LEAST_COST_ROUTING = "LCR"
    RULE_BASED = "RULE"


class Connector(BaseModel):
    """Payment connector model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: ConnectorType
    success_rate: float = Field(ge=0.0, le=1.0)
    cost_per_transaction: float = Field(ge=0.0)
    processing_time_ms: int = Field(ge=0)
    supported_payment_methods: List[PaymentMethod]
    supported_currencies: List[Currency]
    geographic_coverage: List[str]  # Country codes
    volume_capacity: int = Field(ge=0)
    current_volume: int = Field(default=0, ge=0)
    is_active: bool = Field(default=True)
    
    @property
    def utilization_rate(self) -> float:
        """Calculate current utilization rate."""
        if self.volume_capacity == 0:
            return 0.0
        return min(self.current_volume / self.volume_capacity, 1.0)


class Transaction(BaseModel):
    """Transaction model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    merchant_id: str
    amount: float = Field(gt=0)
    currency: Currency
    payment_method: PaymentMethod
    country_code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: TransactionStatus = Field(default=TransactionStatus.PENDING)
    connector_id: Optional[str] = None
    processing_time_ms: Optional[int] = None
    cost: Optional[float] = None
    failure_reason: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)
    
    # Additional metadata
    customer_id: Optional[str] = None
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    is_recurring: bool = Field(default=False)
    merchant_category: Optional[str] = None


class MerchantRule(BaseModel):
    """Merchant-defined routing rule."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    merchant_id: str
    name: str
    priority: int = Field(ge=1)  # Lower number = higher priority
    conditions: Dict[str, Union[str, List[str], float, int]]
    target_connectors: List[str]
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Examples of conditions:
    # {"amount_min": 100, "amount_max": 1000}
    # {"payment_method": ["card", "wallet"]}
    # {"country_code": ["US", "CA"]}
    # {"currency": "USD"}


class VolumeDistribution(BaseModel):
    """Volume distribution configuration."""
    merchant_id: str
    connector_distributions: Dict[str, float]  # connector_id -> percentage (0.0-1.0)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def validate_distributions(self) -> bool:
        """Validate that distributions sum to 1.0."""
        total = sum(self.connector_distributions.values())
        return abs(total - 1.0) < 0.001  # Allow for floating point precision


class RoutingDecision(BaseModel):
    """Routing decision result."""
    transaction_id: str
    selected_connector_id: str
    algorithm_used: List[RoutingAlgorithm]
    confidence_score: float = Field(ge=0.0, le=1.0)
    decision_factors: Dict[str, float]  # Factor name -> weight/score
    alternative_connectors: List[str] = Field(default_factory=list)
    decision_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PerformanceMetrics(BaseModel):
    """Performance metrics for analysis."""
    connector_id: str
    time_period_start: datetime
    time_period_end: datetime
    total_transactions: int = Field(ge=0)
    successful_transactions: int = Field(ge=0)
    failed_transactions: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    average_processing_time_ms: float = Field(ge=0.0)
    total_volume: float = Field(ge=0.0)
    total_cost: float = Field(ge=0.0)
    average_cost_per_transaction: float = Field(ge=0.0)
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate


class OptimizationResult(BaseModel):
    """Result of routing optimization analysis."""
    algorithm_combination: List[RoutingAlgorithm]
    total_transactions: int
    overall_success_rate: float
    total_cost: float
    average_cost_per_transaction: float
    volume_distribution_adherence: float  # How well volume targets were met
    rule_compliance_rate: float  # How well merchant rules were followed
    auth_uplift: float  # Improvement in authorization rate
    cost_reduction: float  # Cost reduction achieved
    performance_score: float  # Overall performance score
    connector_utilization: Dict[str, float]  # connector_id -> utilization rate
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


class RoutingConfiguration(BaseModel):
    """Complete routing configuration for a merchant."""
    merchant_id: str
    enabled_algorithms: List[RoutingAlgorithm]
    algorithm_weights: Dict[RoutingAlgorithm, float]  # Algorithm importance weights
    volume_distribution: Optional[VolumeDistribution] = None
    merchant_rules: List[MerchantRule] = Field(default_factory=list)
    fallback_connector_id: Optional[str] = None
    max_retries: int = Field(default=3, ge=0)
    timeout_ms: int = Field(default=30000, gt=0)
    is_active: bool = Field(default=True)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
