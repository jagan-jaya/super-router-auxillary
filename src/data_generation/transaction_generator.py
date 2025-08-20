"""
Real-time transaction data generator for routing optimization analysis.

This module generates realistic payment transaction data with various patterns
and scenarios to test different routing algorithms and their combinations.
"""

import random
import json
import csv
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from faker import Faker
import numpy as np
import pandas as pd

from ..models import (
    Transaction, Connector, PaymentMethod, Currency, TransactionStatus,
    ConnectorType, MerchantRule, VolumeDistribution
)


class TransactionGenerator:
    """Generates realistic transaction data for testing routing algorithms."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the transaction generator."""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        self.fake = Faker()
        Faker.seed(seed if seed else 0)
        
        # Initialize sample connectors
        self.connectors = self._create_sample_connectors()
        
        # Transaction patterns and distributions
        self.merchant_categories = [
            "e-commerce", "subscription", "marketplace", "gaming", 
            "travel", "food_delivery", "fintech", "saas"
        ]
        
        self.country_distributions = {
            "US": 0.35, "GB": 0.15, "DE": 0.12, "FR": 0.08,
            "IN": 0.10, "CA": 0.06, "AU": 0.05, "JP": 0.04,
            "BR": 0.03, "MX": 0.02
        }
        
        self.payment_method_distributions = {
            PaymentMethod.CARD: 0.65,
            PaymentMethod.WALLET: 0.20,
            PaymentMethod.BANK_TRANSFER: 0.10,
            PaymentMethod.UPI: 0.04,
            PaymentMethod.BNPL: 0.01
        }
        
        # Time-based patterns
        self.hourly_multipliers = self._generate_hourly_patterns()
        self.daily_multipliers = self._generate_daily_patterns()
    
    def _create_sample_connectors(self) -> List[Connector]:
        """Create sample payment connectors with realistic characteristics."""
        connectors = [
            # Global connectors
            Connector(
                name="Stripe",
                type=ConnectorType.GLOBAL,
                success_rate=0.94,
                cost_per_transaction=0.029,  # 2.9%
                processing_time_ms=1200,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.WALLET],
                supported_currencies=[Currency.USD, Currency.EUR, Currency.GBP, Currency.CAD],
                geographic_coverage=["US", "GB", "DE", "FR", "CA", "AU"],
                volume_capacity=100000
            ),
            Connector(
                name="Adyen",
                type=ConnectorType.GLOBAL,
                success_rate=0.92,
                cost_per_transaction=0.028,
                processing_time_ms=1400,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.WALLET, PaymentMethod.BANK_TRANSFER],
                supported_currencies=[Currency.USD, Currency.EUR, Currency.GBP, Currency.JPY],
                geographic_coverage=["US", "GB", "DE", "FR", "JP", "AU"],
                volume_capacity=80000
            ),
            Connector(
                name="PayPal",
                type=ConnectorType.GLOBAL,
                success_rate=0.89,
                cost_per_transaction=0.034,
                processing_time_ms=1800,
                supported_payment_methods=[PaymentMethod.WALLET, PaymentMethod.CARD],
                supported_currencies=[Currency.USD, Currency.EUR, Currency.GBP, Currency.CAD],
                geographic_coverage=["US", "GB", "DE", "FR", "CA", "AU"],
                volume_capacity=60000
            ),
            
            # Regional connectors
            Connector(
                name="Razorpay",
                type=ConnectorType.REGIONAL,
                success_rate=0.91,
                cost_per_transaction=0.022,
                processing_time_ms=900,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.UPI, PaymentMethod.WALLET],
                supported_currencies=[Currency.INR],
                geographic_coverage=["IN"],
                volume_capacity=50000
            ),
            Connector(
                name="Mollie",
                type=ConnectorType.REGIONAL,
                success_rate=0.93,
                cost_per_transaction=0.025,
                processing_time_ms=1100,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.BANK_TRANSFER],
                supported_currencies=[Currency.EUR],
                geographic_coverage=["DE", "FR", "NL", "BE"],
                volume_capacity=40000
            ),
            
            # Local connectors
            Connector(
                name="Square",
                type=ConnectorType.LOCAL,
                success_rate=0.95,
                cost_per_transaction=0.026,
                processing_time_ms=800,
                supported_payment_methods=[PaymentMethod.CARD],
                supported_currencies=[Currency.USD],
                geographic_coverage=["US"],
                volume_capacity=30000
            ),
            Connector(
                name="Worldpay",
                type=ConnectorType.LOCAL,
                success_rate=0.90,
                cost_per_transaction=0.031,
                processing_time_ms=1500,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.WALLET],
                supported_currencies=[Currency.GBP],
                geographic_coverage=["GB"],
                volume_capacity=25000
            ),
            Connector(
                name="Pagseguro",
                type=ConnectorType.LOCAL,
                success_rate=0.88,
                cost_per_transaction=0.035,
                processing_time_ms=2000,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.BANK_TRANSFER],
                supported_currencies=[Currency.USD],  # BRL not in enum, using USD
                geographic_coverage=["BR"],
                volume_capacity=20000
            )
        ]
        
        return connectors
    
    def _generate_hourly_patterns(self) -> List[float]:
        """Generate realistic hourly transaction volume patterns."""
        # Peak hours: 9-11 AM, 2-4 PM, 7-9 PM
        base_pattern = [
            0.3, 0.2, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8,  # 0-7 AM
            1.0, 1.2, 1.1, 0.9, 0.8, 0.9, 1.1, 1.2,  # 8-15 (8AM-3PM)
            1.0, 0.9, 0.8, 1.1, 1.2, 1.0, 0.7, 0.5   # 16-23 (4PM-11PM)
        ]
        return base_pattern
    
    def _generate_daily_patterns(self) -> List[float]:
        """Generate realistic daily transaction volume patterns."""
        # Monday=0, Sunday=6
        # Higher on weekdays, lower on weekends
        return [1.2, 1.3, 1.2, 1.1, 1.0, 0.8, 0.6]  # Mon-Sun
    
    def _get_time_multiplier(self, timestamp: datetime) -> float:
        """Get volume multiplier based on time patterns."""
        hour_mult = self.hourly_multipliers[timestamp.hour]
        day_mult = self.daily_multipliers[timestamp.weekday()]
        return hour_mult * day_mult
    
    def _select_country(self) -> str:
        """Select a country based on realistic distributions."""
        countries = list(self.country_distributions.keys())
        weights = list(self.country_distributions.values())
        return np.random.choice(countries, p=weights)
    
    def _select_payment_method(self, country: str) -> PaymentMethod:
        """Select payment method based on country and distributions."""
        # Adjust distributions based on country
        if country == "IN":
            # Higher UPI usage in India
            methods = [PaymentMethod.CARD, PaymentMethod.UPI, PaymentMethod.WALLET, PaymentMethod.BANK_TRANSFER]
            weights = [0.45, 0.35, 0.15, 0.05]
        elif country in ["DE", "NL", "BE"]:
            # Higher bank transfer usage in Europe
            methods = [PaymentMethod.CARD, PaymentMethod.BANK_TRANSFER, PaymentMethod.WALLET]
            weights = [0.60, 0.25, 0.15]
        else:
            # Default distribution
            methods = list(self.payment_method_distributions.keys())
            weights = list(self.payment_method_distributions.values())
        
        # Use random.choices instead of np.random.choice for enum compatibility
        return random.choices(methods, weights=weights)[0]
    
    def _select_currency(self, country: str) -> Currency:
        """Select currency based on country."""
        currency_map = {
            "US": Currency.USD, "CA": Currency.CAD,
            "GB": Currency.GBP, "DE": Currency.EUR,
            "FR": Currency.EUR, "IN": Currency.INR,
            "JP": Currency.JPY, "AU": Currency.AUD,
            "BR": Currency.USD,  # Using USD as BRL not in enum
            "MX": Currency.USD   # Using USD as MXN not in enum
        }
        return currency_map.get(country, Currency.USD)
    
    def _generate_amount(self, payment_method: PaymentMethod, merchant_category: str) -> float:
        """Generate realistic transaction amounts."""
        # Base amounts by category
        category_ranges = {
            "e-commerce": (10, 500),
            "subscription": (5, 100),
            "marketplace": (20, 1000),
            "gaming": (1, 200),
            "travel": (100, 5000),
            "food_delivery": (15, 150),
            "fintech": (50, 10000),
            "saas": (10, 1000)
        }
        
        min_amt, max_amt = category_ranges.get(merchant_category, (10, 500))
        
        # Adjust for payment method
        if payment_method == PaymentMethod.UPI:
            max_amt = min(max_amt, 200)  # UPI typically for smaller amounts
        elif payment_method == PaymentMethod.BNPL:
            min_amt = max(min_amt, 50)   # BNPL for higher amounts
        
        # Generate with log-normal distribution for realistic skew
        log_min, log_max = np.log(min_amt), np.log(max_amt)
        log_amount = np.random.uniform(log_min, log_max)
        amount = np.exp(log_amount)
        
        return round(amount, 2)
    
    def _generate_risk_score(self, amount: float, payment_method: PaymentMethod, 
                           country: str, is_recurring: bool) -> float:
        """Generate realistic risk scores."""
        base_risk = 0.3
        
        # Amount-based risk
        if amount > 1000:
            base_risk += 0.2
        elif amount > 500:
            base_risk += 0.1
        
        # Payment method risk
        method_risk = {
            PaymentMethod.CARD: 0.0,
            PaymentMethod.WALLET: -0.1,
            PaymentMethod.BANK_TRANSFER: -0.05,
            PaymentMethod.UPI: -0.1,
            PaymentMethod.BNPL: 0.1
        }
        base_risk += method_risk.get(payment_method, 0.0)
        
        # Country risk (simplified)
        high_risk_countries = ["BR", "MX"]
        if country in high_risk_countries:
            base_risk += 0.15
        
        # Recurring transactions are lower risk
        if is_recurring:
            base_risk -= 0.2
        
        # Add some randomness
        base_risk += np.random.normal(0, 0.1)
        
        return max(0.0, min(1.0, base_risk))
    
    def generate_transaction(self, merchant_id: str, timestamp: Optional[datetime] = None) -> Transaction:
        """Generate a single realistic transaction."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        country = self._select_country()
        payment_method = self._select_payment_method(country)
        currency = self._select_currency(country)
        merchant_category = random.choice(self.merchant_categories)
        amount = self._generate_amount(payment_method, merchant_category)
        is_recurring = random.random() < 0.15  # 15% recurring
        
        risk_score = self._generate_risk_score(amount, payment_method, country, is_recurring)
        
        return Transaction(
            merchant_id=merchant_id,
            amount=amount,
            currency=currency,
            payment_method=payment_method,
            country_code=country,
            timestamp=timestamp,
            customer_id=self.fake.uuid4(),
            risk_score=risk_score,
            is_recurring=is_recurring,
            merchant_category=merchant_category
        )
    
    def generate_transaction_stream(self, merchant_id: str, duration_hours: int = 24, 
                                  base_tps: float = 10.0) -> List[Transaction]:
        """Generate a stream of transactions over a time period."""
        transactions = []
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=duration_hours)
        
        current_time = start_time
        while current_time < now:
            # Calculate transactions for this minute
            time_multiplier = self._get_time_multiplier(current_time)
            transactions_per_minute = base_tps * 60 * time_multiplier
            
            # Add some randomness
            actual_transactions = max(0, int(np.random.poisson(transactions_per_minute)))
            
            for _ in range(actual_transactions):
                # Spread transactions across the minute
                transaction_time = current_time + timedelta(
                    seconds=random.uniform(0, 60)
                )
                transaction = self.generate_transaction(merchant_id, transaction_time)
                transactions.append(transaction)
            
            current_time += timedelta(minutes=1)
        
        return sorted(transactions, key=lambda t: t.timestamp)
    
    def generate_multiple_merchants(self, num_merchants: int = 5, 
                                  duration_hours: int = 24) -> Dict[str, List[Transaction]]:
        """Generate transaction data for multiple merchants."""
        merchant_data = {}
        
        for i in range(num_merchants):
            merchant_id = f"merchant_{i+1:03d}"
            # Vary transaction volume by merchant
            base_tps = random.uniform(5.0, 50.0)
            transactions = self.generate_transaction_stream(
                merchant_id, duration_hours, base_tps
            )
            merchant_data[merchant_id] = transactions
        
        return merchant_data
    
    def create_sample_merchant_rules(self, merchant_id: str) -> List[MerchantRule]:
        """Create sample merchant rules for testing."""
        rules = [
            MerchantRule(
                merchant_id=merchant_id,
                name="High Value Transactions",
                priority=1,
                conditions={"amount_min": 1000},
                target_connectors=["stripe", "adyen"]  # Use reliable connectors for high value
            ),
            MerchantRule(
                merchant_id=merchant_id,
                name="India UPI Routing",
                priority=2,
                conditions={"country_code": ["IN"], "payment_method": ["upi"]},
                target_connectors=["razorpay"]
            ),
            MerchantRule(
                merchant_id=merchant_id,
                name="Europe Bank Transfers",
                priority=3,
                conditions={"country_code": ["DE", "FR"], "payment_method": ["bank_transfer"]},
                target_connectors=["mollie", "adyen"]
            ),
            MerchantRule(
                merchant_id=merchant_id,
                name="Low Risk Recurring",
                priority=4,
                conditions={"is_recurring": True, "risk_score_max": 0.3},
                target_connectors=["stripe", "square"]
            )
        ]
        return rules
    
    def create_sample_volume_distribution(self, merchant_id: str) -> VolumeDistribution:
        """Create sample volume distribution configuration."""
        return VolumeDistribution(
            merchant_id=merchant_id,
            connector_distributions={
                "stripe": 0.4,    # 40%
                "adyen": 0.3,     # 30%
                "paypal": 0.15,   # 15%
                "razorpay": 0.1,  # 10%
                "mollie": 0.05    # 5%
            }
        )
    
    def save_to_csv(self, transactions: List[Transaction], filename: str):
        """Save transactions to CSV file."""
        df = pd.DataFrame([t.dict() for t in transactions])
        df.to_csv(filename, index=False)
        print(f"Saved {len(transactions)} transactions to {filename}")
    
    def save_to_json(self, data: Dict, filename: str):
        """Save data to JSON file."""
        # Convert datetime objects to strings for JSON serialization
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filename, 'w') as f:
            json.dump(data, f, default=serialize_datetime, indent=2)
        print(f"Saved data to {filename}")


def main():
    """Main function to generate sample data."""
    print("Generating sample transaction data...")
    
    generator = TransactionGenerator(seed=42)
    
    # Generate data for multiple merchants (reduced for faster execution)
    merchant_data = generator.generate_multiple_merchants(
        num_merchants=3, 
        duration_hours=2  # Reduced from 24 to 2 hours for faster generation
    )
    
    # Create data directory
    import os
    os.makedirs("data", exist_ok=True)
    
    # Save transaction data
    all_transactions = []
    for merchant_id, transactions in merchant_data.items():
        all_transactions.extend(transactions)
        
        # Save individual merchant data
        generator.save_to_csv(
            transactions, 
            f"data/{merchant_id}_transactions.csv"
        )
        
        # Create and save merchant rules
        rules = generator.create_sample_merchant_rules(merchant_id)
        rules_data = [rule.dict() for rule in rules]
        generator.save_to_json(
            rules_data, 
            f"data/{merchant_id}_rules.json"
        )
        
        # Create and save volume distribution
        volume_dist = generator.create_sample_volume_distribution(merchant_id)
        generator.save_to_json(
            volume_dist.dict(), 
            f"data/{merchant_id}_volume_distribution.json"
        )
    
    # Save combined data
    generator.save_to_csv(all_transactions, "data/all_transactions.csv")
    
    # Save connector data
    connectors_data = [connector.dict() for connector in generator.connectors]
    generator.save_to_json(connectors_data, "data/connectors.json")
    
    # Print summary
    print(f"\nData Generation Summary:")
    print(f"Total transactions: {len(all_transactions)}")
    print(f"Merchants: {len(merchant_data)}")
    print(f"Connectors: {len(generator.connectors)}")
    print(f"Time range: {min(t.timestamp for t in all_transactions)} to {max(t.timestamp for t in all_transactions)}")
    
    # Print transaction distribution by merchant
    for merchant_id, transactions in merchant_data.items():
        print(f"{merchant_id}: {len(transactions)} transactions")


if __name__ == "__main__":
    main()
