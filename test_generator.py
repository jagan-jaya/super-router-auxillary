#!/usr/bin/env python3
"""
Simple test script for the transaction generator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_generation.transaction_generator import TransactionGenerator

def main():
    print("Testing transaction generator with minimal data...")
    
    generator = TransactionGenerator(seed=42)
    
    # Generate just a few transactions for testing
    print("Generating 10 sample transactions...")
    transactions = []
    for i in range(10):
        transaction = generator.generate_transaction("test_merchant")
        transactions.append(transaction)
        print(f"Transaction {i+1}: {transaction.amount} {transaction.currency} via {transaction.payment_method} from {transaction.country_code}")
    
    # Test saving to CSV
    print("\nSaving to CSV...")
    generator.save_to_csv(transactions, "test_transactions.csv")
    
    # Test connector data
    print(f"\nAvailable connectors: {len(generator.connectors)}")
    for connector in generator.connectors:
        print(f"- {connector.name} ({connector.type}): {connector.success_rate*100:.1f}% success rate")
    
    # Test merchant rules
    print("\nSample merchant rules:")
    rules = generator.create_sample_merchant_rules("test_merchant")
    for rule in rules:
        print(f"- {rule.name}: {rule.conditions}")
    
    # Test volume distribution
    print("\nSample volume distribution:")
    volume_dist = generator.create_sample_volume_distribution("test_merchant")
    for connector, percentage in volume_dist.connector_distributions.items():
        print(f"- {connector}: {percentage*100:.0f}%")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
