#!/usr/bin/env python3
"""
Script to execute curl commands for all profile_ids in local_data.csv
"""

import csv
import subprocess
import sys
import os
from pathlib import Path

def read_profile_ids(csv_file_path):
    """Read profile_ids from the CSV file"""
    profile_ids = []
    
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                profile_id = row.get('profile_id', '').strip()
                if profile_id:
                    profile_ids.append(profile_id)
        
        print(f"Found {len(profile_ids)} profile IDs in the CSV file")
        return profile_ids
    
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def execute_curl_command(profile_id, api_key):
    """Execute the curl command for a given profile_id"""
    
    # Construct the curl command
    curl_command = [
        'curl',
        '--location',
        'https://sandbox.hyperswitch.io/configs/',
        '--header', 'Content-Type: application/json',
        '--header', 'Accept: application/json',
        '--header', f'api-key: {api_key}',
        '--data', f'{{"key": "routing_result_source_{profile_id}", "value": "decision_engine"}}'
    ]
    
    try:
        print(f"Executing curl for profile_id: {profile_id}")
        
        # Execute the curl command
        result = subprocess.run(
            curl_command,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        # Print the result
        print(f"Profile ID: {profile_id}")
        print(f"Status Code: {result.returncode}")
        print(f"Response: {result.stdout}")
        
        if result.stderr:
            print(f"Error: {result.stderr}")
        
        print("-" * 50)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"Timeout occurred for profile_id: {profile_id}")
        return False
    except Exception as e:
        print(f"Error executing curl for profile_id {profile_id}: {e}")
        return False

def main():
    """Main function"""
    
    # Check if API key is provided
    if len(sys.argv) < 2:
        print("Usage: python execute_curl_for_profiles.py <API_KEY> [CSV_FILE_PATH]")
        print("Example: python execute_curl_for_profiles.py your_api_key_here")
        print("Example: python execute_curl_for_profiles.py your_api_key_here /path/to/local_data.csv")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    # Determine CSV file path
    if len(sys.argv) >= 3:
        csv_file_path = sys.argv[2]
    else:
        # Default paths to try
        possible_paths = [
            "../../Downloads/local_data (2).csv",
            "local_data.csv",
            "local_data (2).csv"
        ]
        
        csv_file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_file_path = path
                break
        
        if not csv_file_path:
            print("Error: Could not find CSV file. Please provide the path as second argument.")
            print("Tried looking for:")
            for path in possible_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    print(f"Using CSV file: {csv_file_path}")
    print(f"API Key: {api_key[:10]}..." if len(api_key) > 10 else f"API Key: {api_key}")
    print("=" * 60)
    
    # Read profile IDs from CSV
    profile_ids = read_profile_ids(csv_file_path)
    
    if not profile_ids:
        print("No profile IDs found. Exiting.")
        sys.exit(1)
    
    # Execute curl commands
    successful_requests = 0
    failed_requests = 0
    
    for i, profile_id in enumerate(profile_ids, 1):
        print(f"\nProcessing {i}/{len(profile_ids)}: {profile_id}")
        
        success = execute_curl_command(profile_id, api_key)
        
        if success:
            successful_requests += 1
        else:
            failed_requests += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total profile IDs processed: {len(profile_ids)}")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    
    if failed_requests > 0:
        print(f"\nWarning: {failed_requests} requests failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\nAll requests completed successfully!")

if __name__ == "__main__":
    main()
