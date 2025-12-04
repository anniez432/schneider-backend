# schneider-backend

## About
This engine generates the top 5 recommended loads for users (referenced by their "USER_PSEUDO_ID" (int) from click-stream(in).csv). If their current/NATNAL location is entered, the engine will prioritize recommending loads closest to that location and can filter by a maximum distance range. If not, the engine relies on their previous location & search history.

The parameters of the engine can be changed as desired. For example, the weights that determine matching users & loads can be changed, load quality doesn't have to be considered, etc.

List of recommended users (most present in clickstream data):
* 1450181150
* 635246794
* 169348607
* 689997252
* 625493898

This engine recommends loads to users based on route similarity (using TF-IDF & cosine similarity), geographic similarity, and load quality (the rpm, distance, etc. of loads).


## Requirements
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```

or use Docker

## Files

### 1. click-stream(in).csv
Provided by Schneider, click stream data collected from the FreightPower app.

### 2. mock_loads.json
Generated when recommendation.py is first ran - can delete and rerun to regenerate mock loads.

### 3. main.py 

### 4. recommendation.py
Generates the mock loads and builds the recommendation engine.

### 5. main.py 
The API server for the engine. Starts FastAPI app and loads the engine.

### 6. recommendations_tester.py
Can be ran to test the recommendations provided for the top 10 most represented users in the click-stream dataset. Can be changed for more comprehensive testing.

### 7. us_cities.csv
The csv used to standardize city/state names between the clickstream csv and recommendation engine.

## Running it!
Can run with uvicorn main:app --reload

which by default means it is available at http://localhost:8000 . Then you can access
* GET /recommend/{user_id} - get recommendations just based on user history
* GET /recommend/{user_id}?current_lat=X&current_lon=Y - get recommendations based on user's current location 
* GET /recommend/{user_id}?current_lat=X&current_lon=Y&max_distance=300 - get recommendations for loads with pickups within specified distance (in miles) from current location
* GET /recommend/{user_id}?desired_date=12/15/2025&desired_time=2:30 PM - get recommendations filtering by user's pickup datetime
* GET /recommend/{user_id}?current_lat=X&current_lon=Y&max_distance=250&desired_date=12/15/2025&desired_time=2:30 PM&limit=10&page=2 - full example, uses user's NATNAL

#### Note: Since we're in the US, make sure to use - (negative) longitude!!

#### Date/Time format:
* **Date Format**: "Dec 3 2025" - uses %b so month must be abbreviated OR "12/3/2025"
* **Time Format**: "2:30 PM" - 12 hour with AM/PM, must be in that format
* Both date & time have to be provided for datetime filtering

Can also run with provided Dockerfile

## Testing
python3 recommendations_tester.py will execute the tester file that tests users with various filter combinations: tests without filters (recommendations purely based on user history), tests with datetime filters, tests with distance range filters, and tests with both distance and datetime filters.

A report is printed for each user & location, and an overall accuracy summary at the end!