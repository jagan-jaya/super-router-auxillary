"""
Enhanced realistic transaction data generator that simulates actual payment processing.

This module generates transaction data that closely matches real-world merchant data
including proper transaction statuses, connector routing, processing times, costs,
and failure scenarios.
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


class RealisticTransactionGenerator:
    """Generates highly realistic transaction data matching real merchant patterns."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the realistic transaction generator."""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        self.fake = Faker()
        Faker.seed(seed if seed else 0)
        
        # Initialize connectors with realistic characteristics
        self.connectors = self._create_realistic_connectors()
        self.connector_map = {c.name.lower(): c for c in self.connectors}
        
        # Real-world transaction patterns
        self.merchant_categories = [
            "e-commerce", "subscription", "marketplace", "gaming", 
            "travel", "food_delivery", "fintech", "saas"
        ]
        
        # Realistic geographic distributions
        self.country_distributions = {
            "US": 0.35, "GB": 0.15, "DE": 0.12, "FR": 0.08,
            "IN": 0.10, "CA": 0.06, "AU": 0.05, "JP": 0.04,
            "BR": 0.03, "MX": 0.02
        }
        
        # Payment method distributions by region
        self.payment_method_distributions = {
            PaymentMethod.CARD: 0.65,
            PaymentMethod.WALLET: 0.20,
            PaymentMethod.BANK_TRANSFER: 0.10,
            PaymentMethod.UPI: 0.04,
            PaymentMethod.BNPL: 0.01
        }
        
        # Realistic failure reasons
        self.failure_reasons = [
            "insufficient_funds",
            "card_declined",
            "expired_card",
            "invalid_cvv",
            "fraud_suspected",
            "network_timeout",
            "issuer_unavailable",
            "limit_exceeded",
            "authentication_failed",
            "invalid_account"
        ]
        
        # Time-based patterns
        self.hourly_multipliers = self._generate_hourly_patterns()
        self.daily_multipliers = self._generate_daily_patterns()
    
    def _create_realistic_connectors(self) -> List[Connector]:
        """Create connectors with realistic success rates and characteristics."""
        connectors = [
            # Global connectors
            Connector(
                id="stripe_global",
                name="Stripe",
                type=ConnectorType.GLOBAL,
                success_rate=0.94,
                cost_per_transaction=0.029,
                processing_time_ms=1200,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.WALLET],
                supported_currencies=[Currency.USD, Currency.EUR, Currency.GBP, Currency.CAD],
                geographic_coverage=["US", "GB", "DE", "FR", "CA", "AU"],
                volume_capacity=100000
            ),
            Connector(
                id="adyen_global",
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
                id="paypal_global",
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
                id="razorpay_in",
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
                id="mollie_eu",
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
                id="square_us",
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
                id="worldpay_gb",
                name="Worldpay",
                type=ConnectorType.LOCAL,
                success_rate=0.90,
                cost_per_transaction=0.031,
                processing_time_ms=1500,
                supported_payment_methods=[PaymentMethod.CARD, PaymentMethod.WALLET],
                supported_currencies=[Currency.GBP],
                geographic_coverage=["GB"],
                volume_capacity=25000
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
        return [1.2, 1.3, 1.2, 1.1, 1.0, 0.8, 0.6]  # Mon-Sun
    
    def _select_connector_for_transaction(self, transaction_data: Dict) -> Connector:
        """Select the most appropriate connector based on transaction characteristics."""
        country = transaction_data['country_code']
        payment_method = transaction_data['payment_method']
        currency = transaction_data['currency']
        amount = transaction_data['amount']
        
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
        
        # Apply routing logic (simplified)
        if country == "IN" and payment_method == PaymentMethod.UPI:
            # Route UPI to Razorpay
            razorpay = next((c for c in compatible_connectors if c.name == "Razorpay"), None)
            if razorpay:
                return razorpay
        
        if amount > 1000:
            # High-value transactions to most reliable connectors
            compatible_connectors.sort(key=lambda x: x.success_rate, reverse=True)
            return compatible_connectors[0]
        
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
    
    def _process_transaction(self, transaction_data: Dict, connector: Connector) -> Dict:
        """Simulate realistic transaction processing with the selected connector."""
        # Determine transaction outcome based on connector success rate and risk factors
        base_success_rate = connector.success_rate
        
        # Adjust success rate based on risk factors
        risk_score = transaction_data['risk_score']
        amount = transaction_data['amount']
        
        # Higher risk = lower success rate
        adjusted_success_rate = base_success_rate * (1 - risk_score * 0.3)
        
        # Very high amounts have slightly lower success rates
        if amount > 5000:
            adjusted_success_rate *= 0.95
        elif amount > 1000:
            adjusted_success_rate *= 0.98
        
        # Determine if transaction succeeds
        is_successful = random.random() < adjusted_success_rate
        
        # Calculate processing time (with some variance)
        base_time = connector.processing_time_ms
        processing_time = int(base_time * random.uniform(0.7, 1.5))
        
        # Calculate actual cost
        cost = amount * connector.cost_per_transaction
        
        if is_successful:
            return {
                'status': TransactionStatus.SUCCESS,
                'connector_id': connector.id,
                'processing_time_ms': processing_time,
                'cost': round(cost, 4),
                'failure_reason': None,
                'retry_count': 0
            }
        else:
            # Transaction failed
            failure_reason = random.choice(self.failure_reasons)
            
            # Some failures might be retried
            retry_count = 0
            if failure_reason in ['network_timeout', 'issuer_unavailable']:
                retry_count = random.randint(1, 3)
            
            return {
                'status': TransactionStatus.FAILED,
                'connector_id': connector.id,
                'processing_time_ms': processing_time,
                'cost': 0.0,  # No cost for failed transactions
                'failure_reason': failure_reason,
                'retry_count': retry_count
            }
    
    def _select_country(self) -> str:
        """Select a country based on realistic distributions."""
        countries = list(self.country_distributions.keys())
        weights = list(self.country_distributions.values())
        return np.random.choice(countries, p=weights)
    
    def _select_payment_method(self, country: str) -> PaymentMethod:
        """Select payment method based on country and distributions."""
        if country == "IN":
            methods = [PaymentMethod.CARD, PaymentMethod.UPI, PaymentMethod.WALLET, PaymentMethod.BANK_TRANSFER]
            weights = [0.45, 0.35, 0.15, 0.05]
        elif country in ["DE", "NL", "BE"]:
            methods = [PaymentMethod.CARD, PaymentMethod.BANK_TRANSFER, PaymentMethod.WALLET]
            weights = [0.60, 0.25, 0.15]
        else:
            methods = list(self.payment_method_distributions.keys())
            weights = list(self.payment_method_distributions.values())
        
        return random.choices(methods, weights=weights)[0]
    
    def _select_currency(self, country: str) -> Currency:
        """Select currency based on country."""
        currency_map = {
            "US": Currency.USD, "CA": Currency.CAD,
            "GB": Currency.GBP, "DE": Currency.EUR,
            "FR": Currency.EUR, "IN": Currency.INR,
            "JP": Currency.JPY, "AU": Currency.AUD,
            "BR": Currency.USD, "MX": Currency.USD
        }
        return currency_map.get(country, Currency.USD)
    
    def _generate_amount(self, payment_method: PaymentMethod, merchant_category: str) -> float:
        """Generate realistic transaction amounts."""
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
        
        if payment_method == PaymentMethod.UPI:
            max_amt = min(max_amt, 200)
        elif payment_method == PaymentMethod.BNPL:
            min_amt = max(min_amt, 50)
        
        # Log-normal distribution for realistic skew
        log_min, log_max = np.log(min_amt), np.log(max_amt)
        log_amount = np.random.uniform(log_min, log_max)
        amount = np.exp(log_amount)
        
        return round(amount, 2)
    
    def _generate_risk_score(self, amount: float, payment_method: PaymentMethod, 
                           country: str, is_recurring: bool) -> float:
        """Generate realistic risk scores."""
        base_risk = 0.3
        
        if amount > 1000:
            base_risk += 0.2
        elif amount > 500:
            base_risk += 0.1
        
        method_risk = {
            PaymentMethod.CARD: 0.0,
            PaymentMethod.WALLET: -0.1,
            PaymentMethod.BANK_TRANSFER: -0.05,
            PaymentMethod.UPI: -0.1,
            PaymentMethod.BNPL: 0.1
        }
        base_risk += method_risk.get(payment_method, 0.0)
        
        high_risk_countries = ["BR", "MX"]
        if country in high_risk_countries:
            base_risk += 0.15
        
        if is_recurring:
            base_risk -= 0.2
        
        base_risk += np.random.normal(0, 0.1)
        return max(0.0, min(1.0, base_risk))
    
    def generate_realistic_transaction(self, merchant_id: str, timestamp: Optional[datetime] = None) -> Transaction:
        """Generate a single realistic transaction with proper processing."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Generate basic transaction data
        country = self._select_country()
        payment_method = self._select_payment_method(country)
        currency = self._select_currency(country)
        merchant_category = random.choice(self.merchant_categories)
        amount = self._generate_amount(payment_method, merchant_category)
        is_recurring = random.random() < 0.15
        
        risk_score = self._generate_risk_score(amount, payment_method, country, is_recurring)
        
        # Create transaction data for connector selection
        transaction_data = {
            'country_code': country,
            'payment_method': payment_method,
            'currency': currency,
            'amount': amount,
            'risk_score': risk_score
        }
        
        # Select appropriate connector
        connector = self._select_connector_for_transaction(transaction_data)
        
        # Process transaction
        processing_result = self._process_transaction(transaction_data, connector)
        
        # Create final transaction
        return Transaction(
            merchant_id=merchant_id,
            amount=amount,
            currency=currency,
            payment_method=payment_method,
            country_code=country,
            timestamp=timestamp,
            status=processing_result['status'],
            connector_id=processing_result['connector_id'],
            processing_time_ms=processing_result['processing_time_ms'],
            cost=processing_result['cost'],
            failure_reason=processing_result['failure_reason'],
            retry_count=processing_result['retry_count'],
            customer_id=self.fake.uuid4(),
            risk_score=risk_score,
            is_recurring=is_recurring,
            merchant_category=merchant_category
        )
    
    def generate_realistic_dataset(self, merchant_id: str, num_transactions: int = 1000) -> List[Transaction]:
        """Generate a realistic dataset with proper transaction distribution."""
        transactions = []
        
        # Generate transactions over the last 24 hours
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=24)
        
        for i in range(num_transactions):
            # Distribute transactions across time period
            time_offset = random.uniform(0, 24 * 60 * 60)  # Random seconds in 24 hours
            transaction_time = start_time + timedelta(seconds=time_offset)
            
            transaction = self.generate_realistic_transaction(merchant_id, transaction_time)
            transactions.append(transaction)
        
        # Sort by timestamp
        transactions.sort(key=lambda t: t.timestamp)
        
        return transactions
    
    def save_to_csv(self, transactions: List[Transaction], filename: str):
        """Save transactions to CSV file."""
        df = pd.DataFrame([t.model_dump() for t in transactions])
        df.to_csv(filename, index=False)
        print(f"Saved {len(transactions)} realistic transactions to {filename}")
    
    def generate_analysis_report(self, transactions: List[Transaction]) -> Dict:
        """Generate analysis report of the transaction data."""
        total_transactions = len(transactions)
        successful_transactions = len([t for t in transactions if t.status == TransactionStatus.SUCCESS])
        failed_transactions = len([t for t in transactions if t.status == TransactionStatus.FAILED])
        
        success_rate = successful_transactions / total_transactions if total_transactions > 0 else 0
        
        # Connector performance
        connector_stats = {}
        for transaction in transactions:
            if transaction.connector_id:
                if transaction.connector_id not in connector_stats:
                    connector_stats[transaction.connector_id] = {
                        'total': 0, 'successful': 0, 'total_cost': 0, 'total_volume': 0
                    }
                
                connector_stats[transaction.connector_id]['total'] += 1
                connector_stats[transaction.connector_id]['total_volume'] += transaction.amount
                
                if transaction.status == TransactionStatus.SUCCESS:
                    connector_stats[transaction.connector_id]['successful'] += 1
                    connector_stats[transaction.connector_id]['total_cost'] += transaction.cost or 0
        
        # Calculate success rates per connector
        for connector_id, stats in connector_stats.items():
            stats['success_rate'] = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            stats['avg_cost_per_transaction'] = stats['total_cost'] / stats['successful'] if stats['successful'] > 0 else 0
        
        return {
            'total_transactions': total_transactions,
            'successful_transactions': successful_transactions,
            'failed_transactions': failed_transactions,
            'overall_success_rate': success_rate,
            'connector_performance': connector_stats,
            'total_volume': sum(t.amount for t in transactions),
            'total_cost': sum(t.cost or 0 for t in transactions if t.status == TransactionStatus.SUCCESS)
        }


def main():
    """Generate realistic transaction data for analysis."""
    print("Generating realistic transaction data...")
    
    generator = RealisticTransactionGenerator(seed=42)
    
    # Generate realistic datasets for multiple merchants
    merchants = ["merchant_001", "merchant_002", "merchant_003"]
    all_transactions = []
    
    for merchant_id in merchants:
        print(f"Generating data for {merchant_id}...")
        
        # Generate different volumes per merchant
        num_transactions = random.randint(800, 1200)
        transactions = generator.generate_realistic_dataset(merchant_id, num_transactions)
        
        # Save individual merchant data
        generator.save_to_csv(transactions, f"data/realistic_{merchant_id}_transactions.csv")
        
        # Generate analysis report
        report = generator.generate_analysis_report(transactions)
        
        print(f"{merchant_id} Summary:")
        print(f"  Total transactions: {report['total_transactions']}")
        print(f"  Success rate: {report['overall_success_rate']:.2%}")
        print(f"  Total volume: ${report['total_volume']:,.2f}")
        print(f"  Total cost: ${report['total_cost']:,.2f}")
        print()
        
        all_transactions.extend(transactions)
    
    # Save combined dataset
    generator.save_to_csv(all_transactions, "data/realistic_all_transactions.csv")
    
    # Generate overall report
    overall_report = generator.generate_analysis_report(all_transactions)
    
    print("Overall Summary:")
    print(f"Total transactions: {overall_report['total_transactions']}")
    print(f"Overall success rate: {overall_report['overall_success_rate']:.2%}")
    print(f"Total volume: ${overall_report['total_volume']:,.2f}")
    print(f"Total processing cost: ${overall_report['total_cost']:,.2f}")
    
    print("\nConnector Performance:")
    for connector_id, stats in overall_report['connector_performance'].items():
        print(f"  {connector_id}:")
        print(f"    Transactions: {stats['total']}")
        print(f"    Success rate: {stats['success_rate']:.2%}")
        print(f"    Volume: ${stats['total_volume']:,.2f}")
        print(f"    Avg cost per transaction: ${stats['avg_cost_per_transaction']:.4f}")


if __name__ == "__main__":
    main()
