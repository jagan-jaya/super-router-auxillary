#!/usr/bin/env python3
"""
Script to check for missing connector implementations in Hyperswitch default_implementations.rs
"""

import re
import sys

def extract_connectors_from_macro(content, macro_name):
    """Extract connector names from a specific macro"""
    pattern = rf'{macro_name}!\(\s*(.*?)\s*\);'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print(f"Macro {macro_name} not found!")
        return set()
    
    connectors_text = match.group(1)
    # Extract connector names
    connector_pattern = r'connectors::(\w+)'
    connectors = set(re.findall(connector_pattern, connectors_text))
    
    return connectors

def find_all_macros(content):
    """Find all macro definitions"""
    macro_pattern = r'macro_rules!\s+(\w+)'
    macros = re.findall(macro_pattern, content)
    return macros

def find_all_macro_calls(content):
    """Find all macro calls"""
    macro_call_pattern = r'(\w+)!\('
    calls = set(re.findall(macro_call_pattern, content))
    return calls

def main():
    try:
        with open('../hyperswitch/crates/hyperswitch_connectors/src/default_implementations.rs', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("Error: default_implementations.rs file not found!")
        return 1

    # Find all macros
    macros = find_all_macros(content)
    print(f"Found {len(macros)} macro definitions:")
    for macro in macros:
        print(f"  - {macro}")
    
    print("\n" + "="*50)
    
    # Find all macro calls
    macro_calls = find_all_macro_calls(content)
    print(f"Found {len(macro_calls)} unique macro calls:")
    for call in sorted(macro_calls):
        print(f"  - {call}")
    
    print("\n" + "="*50)
    
    # Check specific revenue recovery macros
    revenue_macros = [
        'default_imp_for_revenue_recovery_record_back',
        'default_imp_for_billing_connector_payment_sync',
        'default_imp_for_billing_connector_invoice_sync'
    ]
    
    print("Checking revenue recovery macros:")
    
    for macro in revenue_macros:
        print(f"\n--- {macro} ---")
        connectors = extract_connectors_from_macro(content, macro)
        
        if connectors:
            print(f"Found {len(connectors)} connectors:")
            if 'Stripebilling' in connectors:
                print("✅ Stripebilling is included")
            else:
                print("❌ Stripebilling is MISSING!")
            
            # Show first few and last few connectors
            sorted_connectors = sorted(connectors)
            if len(sorted_connectors) <= 10:
                for conn in sorted_connectors:
                    status = "✅" if conn == 'Stripebilling' else "  "
                    print(f"  {status} {conn}")
            else:
                print(f"  First 5: {', '.join(sorted_connectors[:5])}")
                print(f"  Last 5: {', '.join(sorted_connectors[-5:])}")
                if 'Stripebilling' in connectors:
                    print("  ✅ Stripebilling found in list")
        else:
            print("No connectors found or macro not implemented")
    
    # Check if Stripebilling appears in other macros for comparison
    print("\n" + "="*50)
    print("Checking if Stripebilling appears in other macros:")
    
    other_macros = [
        'default_imp_for_authorize_session_token',
        'default_imp_for_revenue_recovery'
    ]
    
    for macro in other_macros:
        connectors = extract_connectors_from_macro(content, macro)
        if 'Stripebilling' in connectors:
            print(f"✅ {macro}: Stripebilling found")
        else:
            print(f"❌ {macro}: Stripebilling missing")

if __name__ == "__main__":
    sys.exit(main())
