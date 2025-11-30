import requests
import sys
import os

# Server is assumed to be running
print("Running tests against existing server...")

try:
    base_url = "http://localhost:8000"
    
    # Test 1: Default pagination (limit=5, page=1)
    print("\nTest 1: Default pagination")
    response = requests.get(f"{base_url}/recommend/1414574370")
    if response.status_code == 200:
        data = response.json()
        count = len(data['recommendations'])
        print(f"Status: {response.status_code}, Count: {count}")
        if count == 5:
            print("PASS: Default limit is 5")
        else:
            print(f"FAIL: Expected 5, got {count}")
    else:
        print(f"FAIL: Status code {response.status_code}")
        print(response.text)

    # Test 2: Custom limit (limit=3)
    print("\nTest 2: Custom limit=3")
    response = requests.get(f"{base_url}/recommend/1414574370?limit=3")
    if response.status_code == 200:
        data = response.json()
        count = len(data['recommendations'])
        print(f"Status: {response.status_code}, Count: {count}")
        if count == 3:
            print("PASS: Limit 3 returned 3 items")
        else:
            print(f"FAIL: Expected 3, got {count}")
    else:
        print(f"FAIL: Status code {response.status_code}")

    # Test 3: Pagination (page 1 vs page 2)
    print("\nTest 3: Pagination (Page 1 vs Page 2)")
    resp1 = requests.get(f"{base_url}/recommend/1414574370?limit=2&page=1")
    resp2 = requests.get(f"{base_url}/recommend/1414574370?limit=2&page=2")
    
    if resp1.status_code == 200 and resp2.status_code == 200:
        data1 = resp1.json()['recommendations']
        data2 = resp2.json()['recommendations']
        
        ids1 = [item['load_id'] for item in data1]
        ids2 = [item['load_id'] for item in data2]
        
        print(f"Page 1 IDs: {ids1}")
        print(f"Page 2 IDs: {ids2}")
        
        if set(ids1).isdisjoint(set(ids2)):
            print("PASS: Page 1 and Page 2 have different items")
        else:
            print("FAIL: Overlap between pages")
            
        # Check ranks
        ranks1 = [item['rank'] for item in data1]
        ranks2 = [item['rank'] for item in data2]
        print(f"Page 1 Ranks: {ranks1}")
        print(f"Page 2 Ranks: {ranks2}")
        
        if ranks1 == [1, 2] and ranks2 == [3, 4]:
             print("PASS: Ranks are correct")
        else:
             print("FAIL: Ranks are incorrect")

    else:
        print("FAIL: API Error")

except Exception as e:
    print(f"An error occurred: {e}")
