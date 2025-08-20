#!/bin/bash

# Script to execute curl commands for all profile_ids in local_data.csv

# Check if API key is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <API_KEY> [CSV_FILE_PATH]"
    echo "Example: $0 your_api_key_here"
    echo "Example: $0 your_api_key_here /path/to/local_data.csv"
    exit 1
fi

API_KEY="$1"

# Determine CSV file path
if [ $# -ge 2 ]; then
    CSV_FILE="$2"
else
    # Try to find the CSV file in common locations
    if [ -f "../../Downloads/local_data (2).csv" ]; then
        CSV_FILE="../../Downloads/local_data (2).csv"
    elif [ -f "local_data.csv" ]; then
        CSV_FILE="local_data.csv"
    elif [ -f "local_data (2).csv" ]; then
        CSV_FILE="local_data (2).csv"
    else
        echo "Error: Could not find CSV file. Please provide the path as second argument."
        echo "Tried looking for:"
        echo "  - ../../Downloads/local_data (2).csv"
        echo "  - local_data.csv"
        echo "  - local_data (2).csv"
        exit 1
    fi
fi

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found at $CSV_FILE"
    exit 1
fi

echo "Using CSV file: $CSV_FILE"
echo "API Key: ${API_KEY:0:10}..."
echo "============================================================"

# Initialize counters
successful_requests=0
failed_requests=0
total_requests=0

# Function to execute curl command
execute_curl() {
    local profile_id="$1"
    local api_key="$2"
    
    echo "Executing curl for profile_id: $profile_id"
    
    # Execute curl command
    response=$(curl --location 'https://sandbox.hyperswitch.io/configs/' \
        --header 'Content-Type: application/json' \
        --header 'Accept: application/json' \
        --header "api-key: $api_key" \
        --data "{\"key\": \"routing_result_source_$profile_id\", \"value\": \"decision_engine\"}" \
        --write-out "HTTPSTATUS:%{http_code}" \
        --silent \
        --max-time 30)
    
    # Extract HTTP status code and response body
    http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    response_body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')
    
    echo "Profile ID: $profile_id"
    echo "HTTP Status: $http_code"
    echo "Response: $response_body"
    
    # Check if request was successful (HTTP 2xx)
    if [[ "$http_code" =~ ^2[0-9][0-9]$ ]]; then
        echo "✓ Success"
        return 0
    else
        echo "✗ Failed"
        return 1
    fi
}

# Read CSV file and process each profile_id
# Skip the header line and extract profile_id (2nd column)
while IFS=',' read -r serial_id profile_id merchant_id active_algorithm_id all_algorithm_ids algorithm_count; do
    # Remove quotes from profile_id if present
    profile_id=$(echo "$profile_id" | sed 's/"//g' | xargs)
    
    # Skip empty profile_ids or header
    if [ -z "$profile_id" ] || [ "$profile_id" = "profile_id" ]; then
        continue
    fi
    
    total_requests=$((total_requests + 1))
    
    echo ""
    echo "Processing $total_requests: $profile_id"
    echo "----------------------------------------"
    
    # Execute curl command
    if execute_curl "$profile_id" "$API_KEY"; then
        successful_requests=$((successful_requests + 1))
    else
        failed_requests=$((failed_requests + 1))
    fi
    
    echo "----------------------------------------"
    
    # Small delay to avoid overwhelming the server
    sleep 0.1
    
done < "$CSV_FILE"

# Summary
echo ""
echo "============================================================"
echo "EXECUTION SUMMARY"
echo "============================================================"
echo "Total profile IDs processed: $total_requests"
echo "Successful requests: $successful_requests"
echo "Failed requests: $failed_requests"

if [ $failed_requests -gt 0 ]; then
    echo ""
    echo "Warning: $failed_requests requests failed. Check the output above for details."
    exit 1
else
    echo ""
    echo "All requests completed successfully!"
fi
