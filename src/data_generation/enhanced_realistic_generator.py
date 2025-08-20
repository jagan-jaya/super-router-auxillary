"""
Enhanced realistic transaction data generator with advanced features:
- Multiple transaction statuses (15+ statuses)
- Time range generation (days, weeks, months, years)
- Artificial downtime simulation
- Real-time parameters matching production systems
- Advanced failure scenarios and recovery patterns
"""

import random
import json
import csv
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Union
from faker import Faker
import numpy as np
import pandas as pd
from enum import Enum

from ..models import (
    Transaction, Connector, PaymentMethod, Currency, TransactionStatus,
    ConnectorType, MerchantRule, VolumeDistribution
)


class TimeRange(str, Enum):
    """Time range options for data generation."""
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"


class DowntimeType(str, Enum):
    """Types of downtime scenarios."""
    MAINTENANCE = "maintenance"
    OUTAGE = "outage"
    DEGRADED = "degraded"
    NETWORK_ISSUE = "network_issue"
    RATE_LIMIT = "rate_limit"


class EnhancedRealisticTransactionGenerator:
    """Enhanced generator with production-like characteristics."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the enhanced realistic transaction generator."""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        self.fake = Faker()
        Faker.seed(seed if seed else 0)
        
        # Initialize connectors with realistic characteristics
        self.connectors = self._create_realistic_connectors()
        self.connector_map = {c.id: c for c in self.connectors}
        
        # Enhanced transaction status distributions
        self.status_distributions = {
            TransactionStatus.SUCCESS: 0.82,
            TransactionStatus.FAILED: 0.08,
            TransactionStatus.DECLINED: 0.04,
            TransactionStatus.TIMEOUT: 0.02,
            TransactionStatus.PENDING: 0.015,
            TransactionStatus.PROCESSING: 0.01,
            TransactionStatus.CANCELLED: 0.008,
            TransactionStatus.AUTHORIZED: 0.005,
            TransactionStatus.CAPTURED: 0.003,
            TransactionStatus.REFUNDED: 0.002,
            TransactionStatus.EXPIRED: 0.002,
            TransactionStatus.VOIDED: 0.001,
            TransactionStatus.DISPUTED: 0.0008,
            TransactionStatus.CHARGEBACK: 0.0005,
            TransactionStatus.PARTIALLY_REFUNDED: 0.0002
        }
        
        # Downtime schedules for connectors
        self.downtime_schedules = self._generate_downtime_schedules()
        
        # Real-world transaction patterns
        self.merchant_categories = [
            "e-commerce", "subscription", "marketplace", "gaming", 
            "travel", "food_delivery", "fintech", "saas", "healthcare",
            "education", "entertainment", "retail", "automotive"
        ]
        
        # Enhanced geographic distributions
        self.country_distributions = {
            "US": 0.32, "GB": 0.14, "DE": 0.11, "FR": 0.08, "IN": 0.09,
            "CA": 0.06, "AU": 0.05, "JP": 0.04, "BR": 0.03, "MX": 0.02,
            "IT": 0.02, "ES": 0.02, "NL": 0.01, "SE": 0.01
        }
        
        # Enhanced failure reasons with categories
        self.failure_reasons = {
            "card_issues": [
                "insufficient_funds", "card_declined", "expired_card", 
                "invalid_cvv", "card_blocked", "invalid_card_number"
            ],
            "fraud_security": [
                "fraud_suspected", "security_violation", "velocity_check_failed",
                "risk_threshold_exceeded", "suspicious_activity"
            ],
            "technical": [
                "network_timeout", "connector_downtime", "system_error",
                "processing_error", "gateway_timeout", "service_unavailable"
            ],
            "issuer": [
                "issuer_unavailable", "issuer_declined", "issuer_timeout",
                "bank_system_error", "authentication_failed"
            ],
            "limits": [
                "limit_exceeded", "daily_limit_reached", "transaction_limit",
                "velocity_limit", "amount_limit_exceeded"
            ]
        }
        
        # Time-based patterns with seasonal variations
        self.hourly_multipliers = self._generate_hourly_patterns()
        self.daily_multipliers = self._generate_daily_patterns()
        self.seasonal_multipliers = self._generate_seasonal_patterns()
        
        # Real-time processing parameters
        self.processing_delays = {
            "card": {"min": 800, "max": 2500, "avg": 1200},
            "wallet": {"min": 500, "max": 1800, "avg": 900},
            "bank_transfer": {"min": 2000, "max": 8000, "avg": 4000},
            "upi": {"min": 300, "max": 1200, "avg": 600},
            "bnpl": {"min": 1500, "max": 5000, "avg": 2500}
        }
    
    def _create_realistic_connectors(self) -> List[Connector]:
        """Create connectors with enhanced realistic characteristics."""
        connectors = [
            # Tier 1 Global connectors (highest reliability)
            Connector(
                id="stripe_global",
                name="Stripe",
                type=ConnectorType.GLOBAL,
                success_rate=0.946,
                cost_per_transaction=0.029,
                processing_time_ms=1200,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.WALLET],
                supported_currencies=[Currency.USD, Currency.EUR, Currency.GBP, Currency.CAD],
                geographic_coverage=["US", "GB", "DE", "FR", "CA", "AU", "NL", "SE"],
                volume_capacity=100000
            ),
            Connector(
                id="adyen_global",
                name="Adyen",
                type=ConnectorType.GLOBAL,
                success_rate=0.938,
                cost_per_transaction=0.028,
                processing_time_ms=1400,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.WALLET, PaymentMethod.BANK_TRANSFER],
                supported_currencies=[Currency.USD, Currency.EUR, Currency.GBP, Currency.JPY],
                geographic_coverage=["US", "GB", "DE", "FR", "JP", "AU", "NL"],
                volume_capacity=80000
            ),
            
            # Tier 2 Global connectors
            Connector(
                id="paypal_global",
                name="PayPal",
                type=ConnectorType.GLOBAL,
                success_rate=0.912,
                cost_per_transaction=0.034,
                processing_time_ms=1800,
                supported_payment_methods=[PaymentMethod.WALLET, PaymentMethod.CARD],
                supported_currencies=[Currency.USD, Currency.EUR, Currency.GBP, Currency.CAD],
                geographic_coverage=["US", "GB", "DE", "FR", "CA", "AU"],
                volume_capacity=60000
            ),
            
            # Regional specialists
            Connector(
                id="razorpay_in",
                name="Razorpay",
                type=ConnectorType.REGIONAL,
                success_rate=0.924,
                cost_per_transaction=0.022,
                processing_time_ms=900,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.UPI, PaymentMethod.WALLET],
                supported_currencies=[Currency.INR],
                geographic_coverage=["IN"],
                volume_capacity=50000
            ),
            Connector(
                id="mollie_eu",
                name="Mollie",
                type=ConnectorType.REGIONAL,
                success_rate=0.931,
                cost_per_transaction=0.025,
                processing_time_ms=1100,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.BANK_TRANSFER],
                supported_currencies=[Currency.EUR],
                geographic_coverage=["DE", "FR", "NL", "BE"],
                volume_capacity=40000
            ),
            
            # Local specialists (highest success rates in their regions)
            Connector(
                id="square_us",
                name="Square",
                type=ConnectorType.LOCAL,
                success_rate=0.958,
                cost_per_transaction=0.026,
                processing_time_ms=800,
                supported_payment_methods=[PaymentMethod.CARD],
                supported_currencies=[Currency.USD],
                geographic_coverage=["US"],
                volume_capacity=30000
            ),
            Connector(
                id="worldpay_gb",
                name="Worldpay",
                type=ConnectorType.LOCAL,
                success_rate=0.918,
                cost_per_transaction=0.031,
                processing_time_ms=1500,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.WALLET],
                supported_currencies=[Currency.GBP],
                geographic_coverage=["GB"],
                volume_capacity=25000
            ),
            
            # Emerging market specialists
            Connector(
                id="payu_latam",
                name="PayU LATAM",
                type=ConnectorType.REGIONAL,
                success_rate=0.885,
                cost_per_transaction=0.035,
                processing_time_ms=2200,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.BANK_TRANSFER],
                supported_currencies=[Currency.USD],
                geographic_coverage=["BR", "MX"],
                volume_capacity=35000
            )
        ]
        
        return connectors
    
    def _generate_downtime_schedules(self) -> Dict[str, List[Dict]]:
        """Generate realistic downtime schedules for connectors with more visible impact."""
        schedules = {}
        
        for connector in self.connectors:
            connector_downtimes = []
            
            # Scheduled maintenance (weekly for more visibility)
            maintenance_day = random.randint(1, 7)  # Day of week
            maintenance_hour = random.randint(2, 5)  # Early morning
            connector_downtimes.append({
                "type": DowntimeType.MAINTENANCE,
                "day_of_week": maintenance_day,
                "hour": maintenance_hour,
                "duration_hours": random.randint(2, 6),  # Longer maintenance windows
                "frequency": "weekly"
            })
            
            # Random outages (more frequent and impactful)
            connector_downtimes.append({
                "type": DowntimeType.OUTAGE,
                "probability_per_day": 0.15,  # 15% chance per day (much higher)
                "duration_hours": random.randint(1, 4),  # 1-4 hour outages
                "frequency": "random"
            })
            
            # Degraded performance periods (more frequent)
            connector_downtimes.append({
                "type": DowntimeType.DEGRADED,
                "probability_per_hour": 0.08,  # 8% chance per hour
                "duration_hours": random.randint(1, 3),  # 1-3 hour degraded periods
                "performance_impact": 0.6,  # 60% performance degradation
                "frequency": "random"
            })
            
            # Peak hour issues (simulate high load failures)
            connector_downtimes.append({
                "type": DowntimeType.RATE_LIMIT,
                "peak_hours": [9, 10, 11, 14, 15, 16, 19, 20, 21],  # Business and evening hours
                "probability_during_peak": 0.25,  # 25% chance during peak hours
                "duration_hours": random.randint(1, 2),
                "performance_impact": 0.8,  # 80% performance degradation
                "frequency": "peak_hours"
            })
            
            schedules[connector.id] = connector_downtimes
        
        return schedules
    
    def _is_connector_down(self, connector_id: str, timestamp: datetime) -> Tuple[bool, Optional[str]]:
        """Check if connector is experiencing downtime at given timestamp with enhanced scenarios."""
        if connector_id not in self.downtime_schedules:
            return False, None
        
        for downtime in self.downtime_schedules[connector_id]:
            if downtime["type"] == DowntimeType.MAINTENANCE:
                # Weekly maintenance windows
                if (timestamp.weekday() == downtime["day_of_week"] and 
                    downtime["hour"] <= timestamp.hour <= downtime["hour"] + downtime["duration_hours"]):
                    return True, "scheduled_maintenance"
            
            elif downtime["type"] == DowntimeType.OUTAGE:
                # Random outages with higher probability
                if random.random() < downtime["probability_per_day"] / 24:
                    return True, "system_outage"
            
            elif downtime["type"] == DowntimeType.DEGRADED:
                # Degraded performance periods
                if random.random() < downtime["probability_per_hour"]:
                    return True, "degraded_performance"
            
            elif downtime["type"] == DowntimeType.RATE_LIMIT:
                # Peak hour issues
                if (timestamp.hour in downtime["peak_hours"] and 
                    random.random() < downtime["probability_during_peak"]):
                    return True, "peak_hour_overload"
        
        return False, None
    
    def _generate_hourly_patterns(self) -> List[float]:
        """Generate realistic hourly transaction volume patterns."""
        # Enhanced patterns with regional variations
        base_pattern = [
            0.25, 0.15, 0.08, 0.06, 0.10, 0.20, 0.40, 0.65,  # 0-7 AM
            0.85, 1.15, 1.25, 1.10, 0.95, 1.05, 1.20, 1.30,  # 8-15 (8AM-3PM)
            1.15, 1.05, 0.95, 1.20, 1.35, 1.10, 0.75, 0.45   # 16-23 (4PM-11PM)
        ]
        return base_pattern
    
    def _generate_daily_patterns(self) -> List[float]:
        """Generate realistic daily transaction volume patterns."""
        return [1.25, 1.35, 1.30, 1.20, 1.15, 0.85, 0.65]  # Mon-Sun
    
    def _generate_seasonal_patterns(self) -> List[float]:
        """Generate seasonal multipliers for each month."""
        # Higher volumes in Nov-Dec (holiday season), lower in Jan-Feb
        return [0.8, 0.75, 0.9, 1.0, 1.05, 1.1, 1.15, 1.1, 1.05, 1.2, 1.4, 1.5]
    
    def _determine_enhanced_transaction_status(self, connector_id: str, risk_score: float, 
                                             amount: float, timestamp: datetime,
                                             payment_method: PaymentMethod) -> Tuple[TransactionStatus, Optional[str], float]:
        """Determine transaction status with enhanced logic including downtime."""
        
        # Check for connector downtime first
        is_down, downtime_reason = self._is_connector_down(connector_id, timestamp)
        if is_down:
            if downtime_reason == "scheduled_maintenance":
                # During maintenance, 90% of transactions fail
                if random.random() < 0.9:
                    return TransactionStatus.TIMEOUT, "scheduled_maintenance", 0.0
            elif downtime_reason == "system_outage":
                # During outages, 95% of transactions fail
                if random.random() < 0.95:
                    return TransactionStatus.FAILED, "connector_unavailable", 0.0
            elif downtime_reason == "degraded_performance":
                # During degraded performance, 60% of transactions fail
                if random.random() < 0.6:
                    return TransactionStatus.TIMEOUT, "slow_response", 0.0
            elif downtime_reason == "peak_hour_overload":
                # During peak hour overload, 80% of transactions fail
                if random.random() < 0.8:
                    return TransactionStatus.FAILED, "rate_limit_exceeded", 0.0
        
        connector = self.connector_map[connector_id]
        base_success_rate = connector.success_rate
        
        # Enhanced risk adjustments
        risk_penalty = risk_score * 0.25
        
        # Amount-based adjustments (more granular)
        if amount > 10000:
            amount_penalty = 0.15
        elif amount > 5000:
            amount_penalty = 0.08
        elif amount > 1000:
            amount_penalty = 0.03
        else:
            amount_penalty = 0
        
        # Time-based adjustments
        hour = timestamp.hour
        time_penalty = 0
        if 22 <= hour or hour <= 6:  # Night hours
            time_penalty = 0.04
        elif 9 <= hour <= 17:  # Business hours - higher load
            time_penalty = 0.02
        
        # Payment method specific adjustments
        method_adjustments = {
            PaymentMethod.CARD: 0.0,
            PaymentMethod.WALLET: 0.02,  # Slightly better
            PaymentMethod.UPI: 0.03,     # Better in supported regions
            PaymentMethod.BANK_TRANSFER: -0.05,  # More prone to issues
            PaymentMethod.BNPL: -0.03    # Additional verification steps
        }
        method_adjustment = method_adjustments.get(payment_method, 0.0)
        
        # Calculate final success rate
        adjusted_success_rate = max(0.1, base_success_rate - risk_penalty - amount_penalty - time_penalty + method_adjustment)
        
        # Determine status with realistic distribution
        rand = random.random()
        cost = amount * connector.cost_per_transaction
        
        if rand < adjusted_success_rate:
            # Success scenarios
            if rand < adjusted_success_rate * 0.95:
                return TransactionStatus.SUCCESS, None, cost
            else:
                # Some successful transactions go through authorization first
                return TransactionStatus.AUTHORIZED, None, cost
        
        else:
            # Failure scenarios with realistic distribution
            failure_rand = random.random()
            
            if failure_rand < 0.4:  # 40% card issues
                reason = random.choice(self.failure_reasons["card_issues"])
                status = TransactionStatus.DECLINED if "declined" in reason else TransactionStatus.FAILED
            elif failure_rand < 0.6:  # 20% fraud/security
                reason = random.choice(self.failure_reasons["fraud_security"])
                status = TransactionStatus.DECLINED
            elif failure_rand < 0.8:  # 20% technical issues
                reason = random.choice(self.failure_reasons["technical"])
                status = TransactionStatus.TIMEOUT if "timeout" in reason else TransactionStatus.FAILED
            elif failure_rand < 0.95:  # 15% issuer issues
                reason = random.choice(self.failure_reasons["issuer"])
                status = TransactionStatus.FAILED
            else:  # 5% limits
                reason = random.choice(self.failure_reasons["limits"])
                status = TransactionStatus.DECLINED
            
            return status, reason, 0.0
    
    def _calculate_realistic_processing_time(self, payment_method: PaymentMethod, 
                                           connector_id: str, status: TransactionStatus,
                                           is_downtime: bool = False) -> int:
        """Calculate realistic processing times based on multiple factors."""
        
        base_times = self.processing_delays.get(payment_method.value, 
                                              self.processing_delays["card"])
        
        # Base processing time with variance
        base_time = np.random.normal(base_times["avg"], base_times["avg"] * 0.2)
        base_time = max(base_times["min"], min(base_times["max"], base_time))
        
        # Connector-specific adjustments
        connector = self.connector_map[connector_id]
        connector_factor = connector.processing_time_ms / 1200  # Normalize to Stripe baseline
        
        # Status-specific adjustments
        status_multipliers = {
            TransactionStatus.SUCCESS: 1.0,
            TransactionStatus.AUTHORIZED: 0.8,
            TransactionStatus.FAILED: 1.3,
            TransactionStatus.DECLINED: 0.9,
            TransactionStatus.TIMEOUT: 3.0,
            TransactionStatus.PROCESSING: 0.5,
            TransactionStatus.PENDING: 0.3
        }
        
        status_multiplier = status_multipliers.get(status, 1.0)
        
        # Downtime impact
        downtime_multiplier = 2.5 if is_downtime else 1.0
        
        final_time = base_time * connector_factor * status_multiplier * downtime_multiplier
        
        return int(max(100, final_time))  # Minimum 100ms
    
    def generate_time_range_dataset(self, merchant_id: str, time_range: TimeRange, 
                                  duration: int, transactions_per_period: int = 1000) -> List[Transaction]:
        """Generate dataset for specified time range."""
        
        # Calculate time period
        end_time = datetime.now(timezone.utc)
        
        if time_range == TimeRange.HOURS:
            start_time = end_time - timedelta(hours=duration)
            total_transactions = transactions_per_period * duration
        elif time_range == TimeRange.DAYS:
            start_time = end_time - timedelta(days=duration)
            total_transactions = transactions_per_period * duration
        elif time_range == TimeRange.WEEKS:
            start_time = end_time - timedelta(weeks=duration)
            total_transactions = transactions_per_period * duration * 7
        elif time_range == TimeRange.MONTHS:
            start_time = end_time - timedelta(days=duration * 30)
            total_transactions = transactions_per_period * duration * 30
        elif time_range == TimeRange.YEARS:
            start_time = end_time - timedelta(days=duration * 365)
            total_transactions = transactions_per_period * duration * 365
        else:
            raise ValueError(f"Unsupported time range: {time_range}")
        
        transactions = []
        
        print(f"Generating {total_transactions} transactions from {start_time} to {end_time}")
        
        for i in range(total_transactions):
            # Distribute transactions across time period with realistic patterns
            time_offset_seconds = random.uniform(0, (end_time - start_time).total_seconds())
            transaction_time = start_time + timedelta(seconds=time_offset_seconds)
            
            # Apply time-based volume adjustments
            hour_multiplier = self.hourly_multipliers[transaction_time.hour]
            day_multiplier = self.daily_multipliers[transaction_time.weekday()]
            month_multiplier = self.seasonal_multipliers[transaction_time.month - 1]
            
            # Skip some transactions based on time patterns (lower volume periods)
            combined_multiplier = hour_multiplier * day_multiplier * month_multiplier
            if random.random() > combined_multiplier / 2.0:  # Normalize to reasonable skip rate
                continue
            
            transaction = self.generate_enhanced_realistic_transaction(merchant_id, transaction_time)
            transactions.append(transaction)
            
            if i % 1000 == 0:
                print(f"Generated {i}/{total_transactions} transactions...")
        
        # Sort by timestamp
        transactions.sort(key=lambda t: t.timestamp)
        
        print(f"Final dataset: {len(transactions)} transactions")
        return transactions
    
    def generate_enhanced_realistic_transaction(self, merchant_id: str, 
                                              timestamp: Optional[datetime] = None) -> Transaction:
        """Generate a single enhanced realistic transaction."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Generate basic transaction data
        country = self._select_country()
        payment_method = self._select_payment_method(country)
        currency = self._select_currency(country)
        merchant_category = random.choice(self.merchant_categories)
        amount = self._generate_amount(payment_method, merchant_category)
        is_recurring = random.random() < 0.18  # Slightly higher for realistic SaaS/subscription
        
        risk_score = self._generate_risk_score(amount, payment_method, country, is_recurring)
        
        # Select appropriate connector
        connector = self._select_connector_for_transaction({
            'country_code': country,
            'payment_method': payment_method,
            'currency': currency,
            'amount': amount,
            'risk_score': risk_score
        })
        
        # Determine transaction status with enhanced logic
        status, failure_reason, cost = self._determine_enhanced_transaction_status(
            connector.id, risk_score, amount, timestamp, payment_method
        )
        
        # Calculate realistic processing time
        is_downtime = failure_reason in ["scheduled_maintenance", "connector_unavailable", "slow_response"]
        processing_time = self._calculate_realistic_processing_time(
            payment_method, connector.id, status, is_downtime
        )
        
        # Determine retry count based on failure type
        retry_count = 0
        if status in [TransactionStatus.TIMEOUT, TransactionStatus.FAILED]:
            if failure_reason in ["network_timeout", "connector_unavailable", "slow_response"]:
                retry_count = random.randint(1, 3)
            elif failure_reason in ["issuer_unavailable", "system_error"]:
                retry_count = random.randint(0, 2)
        
        return Transaction(
            merchant_id=merchant_id,
            amount=amount,
            currency=currency,
            payment_method=payment_method,
            country_code=country,
            timestamp=timestamp,
            status=status,
            connector_id=connector.id,
            processing_time_ms=processing_time,
            cost=round(cost, 4) if cost > 0 else 0.0,
            failure_reason=failure_reason,
            retry_count=retry_count,
            customer_id=self.fake.uuid4(),
            risk_score=round(risk_score, 4),
            is_recurring=is_recurring,
            merchant_category=merchant_category
        )
    
    # Include all the helper methods from the original generator
    def _select_country(self) -> str:
        """Select a country based on realistic distributions."""
        countries = list(self.country_distributions.keys())
        weights = list(self.country_distributions.values())
        return np.random.choice(countries, p=weights)
    
    def _select_payment_method(self, country: str) -> PaymentMethod:
        """Select payment method based on country and distributions."""
        if country == "IN":
            methods = [PaymentMethod.CARD, PaymentMethod.UPI, PaymentMethod.WALLET, PaymentMethod.BANK_TRANSFER]
            weights = [0.42, 0.38, 0.15, 0.05]
        elif country in ["DE", "NL", "BE"]:
            methods = [PaymentMethod.CARD, PaymentMethod.BANK_TRANSFER, PaymentMethod.WALLET]
            weights = [0.58, 0.28, 0.14]
        elif country in ["BR", "MX"]:
            methods = [PaymentMethod.CARD, PaymentMethod.BANK_TRANSFER, PaymentMethod.WALLET]
            weights = [0.65, 0.20, 0.15]
        else:
            methods = [PaymentMethod.CARD, PaymentMethod.WALLET, PaymentMethod.BANK_TRANSFER, PaymentMethod.BNPL]
            weights = [0.68, 0.22, 0.08, 0.02]
        
        return random.choices(methods, weights=weights)[0]
    
    def _select_currency(self, country: str) -> Currency:
        """Select currency based on country."""
        currency_map = {
            "US": Currency.USD, "CA": Currency.CAD, "GB": Currency.GBP,
            "DE": Currency.EUR, "FR": Currency.EUR, "IT": Currency.EUR,
            "ES": Currency.EUR, "NL": Currency.EUR, "IN": Currency.INR,
            "JP": Currency.JPY, "AU": Currency.AUD, "BR": Currency.USD,
            "MX": Currency.USD, "SE": Currency.EUR
        }
        return currency_map.get(country, Currency.USD)
    
    def _generate_amount(self, payment_method: PaymentMethod, merchant_category: str) -> float:
        """Generate realistic transaction amounts with enhanced distributions."""
        category_ranges = {
            "e-commerce": (8, 650),
            "subscription": (4, 120),
            "marketplace": (15, 1200),
            "gaming": (1, 250),
            "travel": (80, 6000),
            "food_delivery": (12, 180),
            "fintech": (25, 15000),
            "saas": (8, 1500),
            "healthcare": (50, 2000),
            "education": (20, 800),
            "entertainment": (5, 300),
            "retail": (10, 800),
            "automotive": (100, 5000)
        }
        
        min_amt, max_amt = category_ranges.get(merchant_category, (10, 500))
        
        # Payment method adjustments
        if payment_method == PaymentMethod.UPI:
            max_amt = min(max_amt, 250)
        elif payment_method == PaymentMethod.BNPL:
            min_amt = max(min_amt, 75)
            max_amt = min(max_amt * 1.5, 3000)
        
        # Use log-normal distribution for realistic skew
        log_min, log_max = np.log(min_amt), np.log(max_amt)
        log_amount = np.random.uniform(log_min, log_max)
        amount = np.exp(log_amount)
        
        return round(amount, 2)
    
    def _generate_risk_score(self, amount: float, payment_method: PaymentMethod, 
                           country: str, is_recurring: bool) -> float:
        """Generate realistic risk scores with enhanced factors."""
        base_risk = 0.25
        
        # Amount-based risk (more granular)
        if amount > 5000:
            base_risk += 0.25
        elif amount > 2000:
            base_risk += 0.15
        elif amount > 1000:
            base_risk += 0.08
        elif amount > 500:
            base_risk += 0.03
        
        # Payment method risk
        method_risk = {
            PaymentMethod.CARD: 0.05,
            PaymentMethod.WALLET: -0.08,
            PaymentMethod.BANK_TRANSFER: -0.03,
            PaymentMethod.UPI: -0.12,
            PaymentMethod.BNPL: 0.12
        }
        base_risk += method_risk.get(payment_method, 0.0)
        
        # Geographic risk
        risk_countries = {
            "BR": 0.18, "MX": 0.15, "IN": 0.08,
            "US": 0.02, "GB": 0.01, "DE": 0.01,
            "CA": 0.02, "AU": 0.03, "JP": 0.01
        }
        base_risk += risk_countries.get(country, 0.05)
        
        # Recurring transaction bonus
        if is_recurring:
            base_risk -= 0.18
        
        # Add some randomness
        base_risk += np.random.normal(0, 0.08)
        
        return max(0.0, min(1.0, base_risk))
    
    def _select_connector_for_transaction(self, transaction_data: Dict) -> Connector:
        """Enhanced connector selection with better routing logic."""
        country = transaction_data['country_code']
        payment_method = transaction_data['payment_method']
        currency = transaction_data['currency']
        amount = transaction_data['amount']
        risk_score = transaction_data['risk_score']
        
        # Filter connectors by compatibility
        compatible_connectors = []
        for connector in self.connectors:
            if (country in connector.geographic_coverage and
                payment_method in connector.supported_payment_methods and
                currency in connector.supported_currencies):
                compatible_connectors.append(connector)
        
        if not compatible_connectors:
            # Fallback to global connectors
            compatible_connectors = [c for c in self.connectors if c.type == ConnectorType.GLOBAL]
        
        # Enhanced routing logic
        if country == "IN" and payment_method == PaymentMethod.UPI:
            # Route UPI to Razorpay
            razorpay = next((c for c in compatible_connectors if c.name == "Razorpay"), None)
            if razorpay:
                return razorpay
        
        if amount > 5000:
            # High-value transactions to most reliable connectors
            compatible_connectors.sort(key=lambda x: x.success_rate, reverse=True)
            return compatible_connectors[0]
        
        if risk_score > 0.7:
            # High-risk transactions to specialized connectors
            local_connectors = [c for c in compatible_connectors if c.type == ConnectorType.LOCAL]
            if local_connectors:
                return max(local_connectors, key=lambda x: x.success_rate)
        
        # Default: weighted selection based on success rate and cost
        weights = []
        for connector in compatible_connectors:
            # Higher success rate and lower cost = higher weight
            weight = connector.success_rate * (1 / (connector.cost_per_transaction + 0.01))
            weights.append(weight)
        
        if weights:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            return random.choices(compatible_connectors, weights=weights)[0]
        
        return compatible_connectors[0] if compatible_connectors else self.connectors[0]
    
    def save_to_csv(self, transactions: List[Transaction], filename: str):
        """Save transactions to CSV file."""
        df = pd.DataFrame([t.model_dump() for t in transactions])
        df.to_csv(filename, index=False)
        print(f"Saved {len(transactions)} enhanced realistic transactions to {filename}")
    
    def generate_analysis_report(self, transactions: List[Transaction]) -> Dict:
        """Generate enhanced analysis report of the transaction data."""
        total_transactions = len(transactions)
        
        # Status distribution
        status_counts = {}
        for status in TransactionStatus:
            status_counts[status.value] = len([t for t in transactions if t.status == status])
        
        successful_transactions = status_counts.get('success', 0)
        failed_transactions = sum(status_counts[status] for status in ['failed', 'declined', 'timeout'])
        
        success_rate = successful_transactions / total_transactions if total_transactions > 0 else 0
        
        # Connector performance
        connector_stats = {}
        for transaction in transactions:
            if transaction.connector_id:
                if transaction.connector_id not in connector_stats:
                    connector_stats[transaction.connector_id] = {
                        'total': 0, 'successful': 0, 'total_cost': 0, 
                        'total_volume': 0, 'avg_processing_time': 0,
                        'downtime_incidents': 0
                    }
                
                stats = connector_stats[transaction.connector_id]
                stats['total'] += 1
                stats['total_volume'] += transaction.amount
                stats['avg_processing_time'] += transaction.processing_time_ms or 0
                
                if transaction.status == TransactionStatus.SUCCESS:
                    stats['successful'] += 1
                    stats['total_cost'] += transaction.cost or 0
                
                if transaction.failure_reason in ["scheduled_maintenance", "connector_unavailable", "slow_response"]:
                    stats['downtime_incidents'] += 1
        
        # Calculate averages and rates
        for connector_id, stats in connector_stats.items():
            stats['success_rate'] = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            stats['avg_cost_per_transaction'] = stats['total_cost'] / stats['successful'] if stats['successful'] > 0 else 0
            stats['avg_processing_time'] = stats['avg_processing_time'] / stats['total'] if stats['total'] > 0 else 0
            stats['downtime_rate'] = stats['downtime_incidents'] / stats['total'] if stats['total'] > 0 else 0
        
        return {
            'total_transactions': total_transactions,
            'status_distribution': status_counts,
            'overall_success_rate': success_rate,
            'connector_performance': connector_stats,
            'total_volume': sum(t.amount for t in transactions),
            'total_cost': sum(t.cost or 0 for t in transactions if t.status == TransactionStatus.SUCCESS),
            'avg_processing_time': sum(t.processing_time_ms or 0 for t in transactions) / total_transactions if total_transactions > 0 else 0,
            'failure_reasons': self._analyze_failure_reasons(transactions),
            'geographic_distribution': self._analyze_geographic_distribution(transactions),
            'payment_method_distribution': self._analyze_payment_method_distribution(transactions)
        }
    
    def _analyze_failure_reasons(self, transactions: List[Transaction]) -> Dict:
        """Analyze failure reasons distribution."""
        failure_reasons = {}
        failed_transactions = [t for t in transactions if t.failure_reason]
        
        for transaction in failed_transactions:
            reason = transaction.failure_reason
            if reason not in failure_reasons:
                failure_reasons[reason] = 0
            failure_reasons[reason] += 1
        
        return failure_reasons
    
    def _analyze_geographic_distribution(self, transactions: List[Transaction]) -> Dict:
        """Analyze geographic distribution of transactions."""
        geo_stats = {}
        for transaction in transactions:
            country = transaction.country_code
            if country not in geo_stats:
                geo_stats[country] = {'count': 0, 'volume': 0, 'success_rate': 0}
            
            geo_stats[country]['count'] += 1
            geo_stats[country]['volume'] += transaction.amount
        
        # Calculate success rates
        for country in geo_stats:
            country_transactions = [t for t in transactions if t.country_code == country]
            successful = len([t for t in country_transactions if t.status == TransactionStatus.SUCCESS])
            geo_stats[country]['success_rate'] = successful / len(country_transactions) if country_transactions else 0
        
        return geo_stats
    
    def _analyze_payment_method_distribution(self, transactions: List[Transaction]) -> Dict:
        """Analyze payment method distribution and performance."""
        method_stats = {}
        for transaction in transactions:
            method = transaction.payment_method.value
            if method not in method_stats:
                method_stats[method] = {'count': 0, 'volume': 0, 'success_rate': 0, 'avg_processing_time': 0}
            
            method_stats[method]['count'] += 1
            method_stats[method]['volume'] += transaction.amount
            method_stats[method]['avg_processing_time'] += transaction.processing_time_ms or 0
        
        # Calculate rates and averages
        for method in method_stats:
            method_transactions = [t for t in transactions if t.payment_method.value == method]
            successful = len([t for t in method_transactions if t.status == TransactionStatus.SUCCESS])
            method_stats[method]['success_rate'] = successful / len(method_transactions) if method_transactions else 0
            method_stats[method]['avg_processing_time'] = method_stats[method]['avg_processing_time'] / len(method_transactions) if method_transactions else 0
        
        return method_stats
