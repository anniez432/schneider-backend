# schneider-backend

## About
This engine generates the top 5 recommended loads for users (referenced by their "USER_PSEUDO_ID" (int) from click-stream(in).csv). If their current/NATNAL location is entered, the engine will prioritize recommending loads closest to that location. If not, the engine relies on their previous location & search history.

The parameters of the engine can be changed as desired. For example, the weights that determine matching users & loads can be changed, load quality doesn't have to be considered, etc.

## Requirements
pip install -r requirements.txt 

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

### us_cities.csv
The csv used to standardize city/state names between the clickstream csv and recommendation engine.

## Running it!
Can run with uvicorn main:app --reload

which by default means it is available at http://localhost:8000 . Then you can access GET /recommend/{user_id} or GET /recommend/{user_id}?current_lat=X&current_lon=Y .

### Note: 
Since we're in the US, make sure to use - (negative) longitude!!

Can also run with provided Dockerfile