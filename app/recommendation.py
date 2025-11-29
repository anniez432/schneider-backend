import json
import random
import pandas as pd
import numpy as np
import pgeocode
import re
from math import radians, sin, cos, sqrt, atan2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# generate the mock loads data
def generate_mock_loads(num_loads = 700, save_path = "mock_loads.json"):
    # generate loads that are biased towards major trucking areas
    trucking_hubs = {
        # Texas
        'TX': {
            'weight': 0.20,
            'cities': ['Houston', 'Dallas', 'Austin', 'San Antonio', 'Fort Worth', 'El Paso', 'Laredo', 'McAllen', 'Waco', 'Amarillo']
        },
        # California
        'CA': {
            'weight': 0.15,
            'cities': ['Los Angeles', 'San Diego', 'San Francisco', 'Sacramento', 'San Jose', 'Fresno', 'Bakersfield', 'Stockton', 'Oakland', 'Long Beach']
        },
        # Illinois
        'IL': {
            'weight': 0.10,
            'cities': ['Chicago', 'Springfield', 'Aurora', 'Naperville', 'Joliet', 'Rockford', 'Peoria', 'Elgin']
        },
        # Florida
        'FL': {
            'weight': 0.08,
            'cities': ['Miami', 'Orlando', 'Tampa', 'Jacksonville', 'Tallahassee', 'Lakeland', 'Ocala', 'Fort Lauderdale', 'Port St. Lucie']
        },
        # Georgia
        'GA': {
            'weight': 0.08,
            'cities': ['Atlanta', 'Savannah', 'Augusta', 'Macon', 'Columbus', 'Athens']
        },
        # Pennsylvania
        'PA': {
            'weight': 0.07,
            'cities': ['Philadelphia', 'Pittsburgh', 'Allentown', 'Harrisburg', 'Erie', 'Lancaster']
        },
        # New York
        'NY': {
            'weight': 0.07,
            'cities': ['New York', 'Buffalo', 'Rochester', 'Albany', 'Syracuse', 'Yonkers', 'Binghamton']
        },
        # Ohio
        'OH': {
            'weight': 0.06,
            'cities': ['Columbus', 'Cleveland', 'Cincinnati', 'Toledo', 'Akron', 'Dayton', 'Youngstown']
        },
        # Missouri
        'MO': {
            'weight': 0.06,
            'cities': ['Kansas City', 'St. Louis', 'Springfield', 'Columbia', 'Independence', 'Joplin']
        },
        # Tennessee
        'TN': {
            'weight': 0.06,
            'cities': ['Nashville', 'Memphis', 'Knoxville', 'Chattanooga', 'Clarksville', 'Murfreesboro']
        },
        # Colorado
        'CO': {
            'weight': 0.04,
            'cities': ['Denver', 'Boulder', 'Colorado Springs', 'Fort Collins', 'Pueblo', 'Greeley']
        },
        # Washington
        'WA': {
            'weight': 0.04,
            'cities': ['Seattle', 'Spokane', 'Tacoma', 'Vancouver', 'Kent', 'Bellevue']
        },
        # Arizona
        'AZ': {
            'weight': 0.03,
            'cities': ['Phoenix', 'Tucson', 'Mesa', 'Scottsdale', 'Flagstaff', 'Yuma', 'Nogales']
        },
        # Michigan
        'MI': {
            'weight': 0.03,
            'cities': ['Detroit', 'Grand Rapids', 'Ann Arbor', 'Lansing', 'Flint', 'Warren']
        },
        # Minnesota
        'MN': {
            'weight': 0.03,
            'cities': ['Minneapolis', 'Saint Paul', 'Rochester', 'Duluth', 'Bloomington']
        },
        # North Carolina
        'NC': {
            'weight': 0.03,
            'cities': ['Charlotte', 'Raleigh', 'Greensboro', 'Durham', 'Fayetteville', 'Winston-Salem', 'Wilmington']
        },
        'AL': {'weight': 0.01, 'cities': ['Birmingham', 'Montgomery', 'Mobile', 'Huntsville']},
        'AR': {'weight': 0.01, 'cities': ['Little Rock', 'Fayetteville', 'Fort Smith', 'Springdale']},
        'CT': {'weight': 0.01, 'cities': ['Bridgeport', 'New Haven', 'Hartford', 'Stamford']},
        'IN': {'weight': 0.01, 'cities': ['Indianapolis', 'Fort Wayne', 'Evansville', 'South Bend']},
        'KS': {'weight': 0.01, 'cities': ['Wichita', 'Overland Park', 'Topeka', 'Kansas City']},
        'KY': {'weight': 0.01, 'cities': ['Louisville', 'Lexington', 'Bowling Green', 'Frankfort']},
        'LA': {'weight': 0.01, 'cities': ['New Orleans', 'Baton Rouge', 'Shreveport', 'Lafayette']},
        'MA': {'weight': 0.01, 'cities': ['Boston', 'Worcester', 'Springfield', 'Cambridge']},
        'NJ': {'weight': 0.01, 'cities': ['Newark', 'Jersey City', 'Paterson', 'Trenton']},
        'NV': {'weight': 0.01, 'cities': ['Las Vegas', 'Reno', 'Henderson', 'North Las Vegas']},
        'OR': {'weight': 0.01, 'cities': ['Portland', 'Salem', 'Eugene', 'Gresham']},
        'SC': {'weight': 0.01, 'cities': ['Columbia', 'Charleston', 'Greenville', 'Spartanburg']},
        'UT': {'weight': 0.01, 'cities': ['Salt Lake City', 'Provo', 'Ogden', 'St. George']},
        'VA': {'weight': 0.01, 'cities': ['Virginia Beach', 'Richmond', 'Norfolk', 'Arlington']},
        'WI': {'weight': 0.01, 'cities': ['Milwaukee', 'Madison', 'Green Bay', 'Kenosha']},
        'DC': {'weight': 0.005, 'cities': ['Washington']},
    }

    states = []
    weights = []
    
    for state, data in trucking_hubs.items():
        states.append(state)
        weights.append(data["weight"])

    # normalize
    weights = [weight / sum(weights) for weight in weights]

    mock_loads = []

    for i in range(num_loads):
        pickup_state = np.random.choice(states, p=weights, size = 1)[0]
        delivery_state = np.random.choice(states, p=weights, size = 1)[0]

        pickup_city = random.choice(trucking_hubs[pickup_state]["cities"])
        delivery_city = random.choice(trucking_hubs[delivery_state]["cities"])

        # can be in the same state but not same city
        while delivery_city == pickup_city and pickup_state == delivery_state:
            delivery_city = random.choice(trucking_hubs[delivery_state]["cities"])

        pickup_time = f'{random.randint(1, 12)}:{random.randint(0,59):02d} {"AM" if random.randint(0,1)==0 else "PM"}'
        delivery_time = f'{random.randint(1, 12)}:{random.randint(0,59):02d} {"AM" if random.randint(0,1)==0 else "PM"}'

        load = {
            'id': str(i + 1),
            'price': random.randint(100, 3000),
            'distance': round(random.uniform(50, 2500), 1),
            'weight': random.randint(5000, 45000),
            'loadedRPM': round(random.uniform(0.5, 2.5), 2),
            'estTotalRPM': round(random.uniform(0.2, 2.0), 2),
            'pickup': {
                'city': pickup_city,
                'state': pickup_state,
                'date': f'Nov {random.randint(1, 20)}',
                'time': pickup_time,
                'emptyMiles': random.randint(0, 300),
                'address': f'{random.randint(1,1000)} Main St',
                'liveLoad': random.choice([True, False])
            },
            'delivery': {
                'city': delivery_city,
                'state': delivery_state,
                'date': f'Nov {random.randint(21, 30)}',
                'time': delivery_time,
                'emptyMiles': random.randint(0, 150),
                'address': f'{random.randint(1,1000)} Warehouse Blvd',
                'instructions': random.sample(['Handle with care', 'Call before arrival', 'Drop at dock', 'Delivery at dock 5', 'Appointment required'], k=2)
            },
            'isReload': random.choice([True, None]),
            'badge': random.choice(['!', None])
        }
         
        # add this mock load
        mock_loads.append(load)

    # save to JSON file @ specified save path
    with open(save_path, "w") as f:
        json.dump(mock_loads, f, indent=2)

    #print('Saved 200 mock loads to mock_loads.json')
    return mock_loads


# geocoding & calculating distances
geo = pgeocode.Nominatim('US')
city_cache = {}

# formatting
def normalize_city(city):
    if not isinstance(city, str):
        return ""
    city = city.upper().strip()
    # remove puncutation
    city = re.sub(r"[^\w\s]", "", city) 
    city = re.sub(r"\s+(TOWNSHIP|CITY|ESTATES|NORTH|SOUTH|EAST|WEST)$", "", city)
    return city

# get latitutde & longitude
def get_latlon(city_state):
    if city_state in city_cache:
        return city_cache[city_state]

    if not isinstance(city_state, str) or ',' not in city_state:
        city_cache[city_state] = None
        return None

    # normalize
    city, state = city_state.split(',')
    city = normalize_city(city)
    state = state.strip().upper()

    # ignore clearly invalid cities
    if city in ["NAN", "", "ANYWHERE_"] or state == "":
        city_cache[city_state] = None
        return None

    # filter pgeocode
    df_geo_data = geo._data
    match = df_geo_data[(df_geo_data['place_name'].str.upper() == city) & (df_geo_data['state_code'] == state)]

    # if there's no match
    if len(match) == 0:
        city_cache[city_state] = None
        return None

    # if there's a match get latitude & longitude
    latitude = match.iloc[0]['latitude']
    longitude = match.iloc[0]['longitude']
    coord = (latitude, longitude)
    city_cache[city_state] = coord
    return coord

# compute the distances between the 2 coordinates in miles
def haversine(coord1, coord2):
    if coord1 is None or coord2 is None:
        return float("inf")

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 3958.8  # miles

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c

# stdize the distance to a similarity score
def distance_to_similarity(dist):
    if dist == float("inf"):
        return 0
    return np.exp(-dist / 300.0)

# use the fallback coordinates ("center of the US") if the coordinates are invalid
def fix_coord(coord):
    fallback_coord = (39.8283, -98.5795)
    if not isinstance(coord, (list, tuple)):
        return fallback_coord
    if len(coord) != 2:
        return fallback_coord

    latitude, longitude = coord
    if pd.isna(latitude) or pd.isna(longitude) or np.isinf(latitude) or np.isinf(longitude):
        return fallback_coord

    return (float(latitude), float(longitude))

# calculate load quality - just based off the mock load info
# can remove this critera!
def calculate_load_quality(loads_df):
    # this will be the same for all users
    loads_df = loads_df.copy()

    # get the total miles that would be traveled
    loads_df["total_miles"] = loads_df["distance"] + loads_df["pickup"].apply(lambda x: x["emptyMiles"]) + \
                              loads_df["delivery"].apply(lambda x: x["emptyMiles"])

    # create revenue/mile statistic
    # + 1 to distance in case distance is 0 for some reason
    loads_df["revenue_per_mile"] = loads_df["price"] / (loads_df["total_miles"] + 1)

    # normalizing to max load weight of 45,000 lbs because of dummy data 
    loads_df["weight_efficiency"] = (loads_df["weight"] - 5000) / 40000
    loads_df["weight_efficiency"] = loads_df["weight_efficiency"].clip(0,1)

    # rev per mile quality
    loads_df["rpm_quality"] = loads_df["loadedRPM"] / (loads_df["estTotalRPM"] + 0.1)
    loads_df["rpm_quality"] = loads_df["rpm_quality"].clip(0,2)

    # normalize dynamiclaly
    rpm_min = loads_df["revenue_per_mile"].min()
    rpm_max = loads_df["revenue_per_mile"].max()
    if rpm_max - rpm_min > 0:
        rpm_normalized = (loads_df["revenue_per_mile"] - rpm_min) / (rpm_max - rpm_min)
    # if all the values are equal
    else:
        rpm_normalized = 0.5 

    # total quality score & normalized
    # revenue per mile is most important
    # then weight efficiency since it's how much of the truck's capacity is used
    # and rpm quality is last because it's based off of performance 
    loads_df["load_quality"] = (
        (0.5 * rpm_normalized) + (0.3 * loads_df["weight_efficiency"]) + (0.2 * (loads_df["rpm_quality"] / 2)).clip(0, 1))
    
    return loads_df

# now want the load quality personalized to the user
def calculate_personalized_load_quality(loads_df, user_profile):
    loads_df = loads_df.copy()

    # get the user info with fallbacks if needed
    home_coordinates = user_profile.get("home_coordinates", (39.8283, -98.5795))
    deadhead_sensitivity = user_profile.get("deadhead_sensitivity", 0.5)
    revenue_priority = user_profile.get("revenue_priority", 0.5)
    consistency_priority = user_profile.get("consistency_priority", 0.2)
    user_avg_rpm = user_profile.get("avg_user_revenue_per_mile", 1.8)
    is_anywhere_searcher = user_profile.get("is_anywhere_searcher", False)

    # deadheads should have penaltieis
    if "pickup_coord" in loads_df.columns:
        deadhead_distances = []
        for pickup_coordinates in loads_df["pickup_coord"]:
            distance = haversine(home_coordinates, pickup_coordinates)
            deadhead_distances.append(distance)
        
        deadhead_normalized = np.array(deadhead_distances) / max(deadhead_distances + [1])
        deadhead_penalty = deadhead_sensitivity * deadhead_normalized
    else:
        deadhead_penalty = 0

    # revenue alignment: how well does the load's rev/mi match user's rate
    revenue_alignment = 1 - np.abs((loads_df["revenue_per_mile"] - user_avg_rpm) / (user_avg_rpm + 0.5)).clip(0, 1) * 0.3 

    # if loads have good rev/mi quality that's good
    rpm_consistency = loads_df["rpm_quality"].clip(0,1)

    # each user's personal quality score
    base_quality = loads_df["load_quality"]

    # user's personal quality score is most important
    # then the revenue's score
    # then the consistency of the rev/mi & deadhead penalty
    # normalized
    personalized_quality_score = ((base_quality * 0.6) + 
                                  (revenue_alignment * revenue_priority * 0.2) +
                                  (rpm_consistency * consistency_priority * 0.1) -
                                  (deadhead_penalty * 0.1)).clip(0, 1)
    
    loads_df["personalized_load_quality"] = personalized_quality_score

    return loads_df

# use their search history to determine this user's preferences
def infer_user_profile(df_model, user_index):
    user_row = df_model.iloc[user_index]

    activity = user_row["user_activity_score"]
    diversity = user_row["search_diversity"]
    anywhere_ratio = user_row.get("anywhere_ratio", 0.0)

    revenue_priority = 0.5 + (activity * 0.3)
    consistency_priority = 1 - revenue_priority + 0.2

    deadhead_sensitivity = 1 - diversity

    if anywhere_ratio > 0.3:
        deadhead_sensitivity = deadhead_sensitivity * 0.5

    # return the user profile dict
    return {
        "home_coordinates": user_row["user_coordinates"],
        "deadhead_sensitivity": float(deadhead_sensitivity),
        "revenue_priority": float(revenue_priority),
        "consistency_priority": float(consistency_priority),
        "avg_user_revenue_per_mile": 1.8,
        "is_anywhere_searcher": anywhere_ratio > 0.3
    }

# data preprocessing

def normalize_geo_to_event(df, mapping_file="us_cities.csv"):
    
    state_name_to_abbr = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
        'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
        'DISTRICT OF COLUMBIA': 'DC',  # ADD THIS LINE
        'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
        'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
        'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
        'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
        'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
        'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
        'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI',
        'SOUTH CAROLINA': 'SC', 'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN',
        'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT', 'VIRGINIA': 'VA',
        'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY'
    }

    df = df.copy()

    if 'GEO_CITY' not in df.columns or 'GEO_REGION' not in df.columns:
        print("ERROR: GEO_CITY or GEO_REGION columns not found")
        df['GEO_CITY_STANDARDIZED'] = "ANYWHERE"
        return df

    # set the copied df's city and region to match the format
    df['GEO_CITY'] = df['GEO_CITY'].fillna('').astype(str).str.strip().str.upper()
    df['GEO_REGION'] = df['GEO_REGION'].fillna('').astype(str).str.strip().str.upper()

    def standardize_row(city, region):
        # want to have them all in city, state_code format
        if not city or len(city) < 2:
            return "ANYWHERE"
        if not region or len(region) < 2:
            return "ANYWHERE"
        
        if city in ['ANYWHERE', 'ANY', 'NATIONWIDE']:
            return "ANYWHERE"
        if region in ['ANYWHERE', 'ANY', 'NATIONWIDE']:
            return "ANYWHERE"
        
        if city in ['NAN', 'UNKNOWN', 'FIREBASE']:
            return "ANYWHERE"
        if region in ['NAN', 'UNKNOWN', 'FIREBASE']:
            return "ANYWHERE"
        
        state_abbr = state_name_to_abbr.get(region, None)

        if state_abbr is None:
            if len(region) == 2:
                state_abbr = region
            else:
                return "ANYWHERE"
            
        if not state_abbr or len(state_abbr) != 2:
            return "ANYWHERE"

        return f"{city},{state_abbr}"


    """
    def to_event_format(row):
      # need to change the format of the city/region
        city = row['GEO_CITY']
        region = row['GEO_REGION']

        # if city is in present get the possible states the city could be in
        if city in grouped:
            possible_states = grouped[city]

            # if the region (og df) is in the abbreviated states list then we have the city/state combo
            if region in state_name_to_abbr:
                state_abbr = state_name_to_abbr[region]
                if state_abbr in possible_states:
                    return f"{city},{state_abbr}"

            # we can just pick the first one - AREA FOR IMPROVEMENT
            return f"{city},{possible_states[0]}"

        # and then if the city isn't present we can just add with the format
        if region in state_name_to_abbr:
            return f"{city},{state_name_to_abbr[region]}"

        # if nothing else works default event_destination format
        return "Anywhere_"

    # add the column with the formatted city, states
    df['GEO_CITY_STANDARDIZED'] = df.apply(to_event_format, axis=1)

    return df
    """
    df['GEO_CITY_STANDARDIZED'] = df.apply(
        lambda row: standardize_row(row['GEO_CITY'], row['GEO_REGION']),
        axis=1
    )

    return df

# get the previous certain # of searches from the user, only top 3 are important
def get_prev_searches(df, n_features = 3):
  df = df.copy()
  """
  for i in range(1, n_features + 1):
    df[f'PREV_SEARCH_{i}'] = df.groupby('USER_PSEUDO_ID')['EVENT_DESTINATION'].shift(i)

  # get those locations too
  for i in range(1, n_features + 1):
    df[f'PREV_SEARCH_LOCATION_{i}'] = df.groupby('USER_PSEUDO_ID')['GEO_CITY_STANDARDIZED'].shift(i)
  """

  # get the previous origins
  for i in range(1, n_features + 1):
    df[f'PREV_ORIGIN_{i}'] = df.groupby('USER_PSEUDO_ID')['EVENT_ORIGIN'].shift(i)

  """
  # get the previous device locations
  for i in range(1, n_features + 1):
    df[f'PREV_GEO_CTIY{i}'] = df.groupby('USER_PSEUDO_ID')['GEO_CITY_STANDARDIZED'].shift(i)
    df[f'PREV_GEO_REGION{i}'] = df.groupby('USER_PSEUDO_ID')['GEO_REGION'].shift(i)

  # get time between each search in hours
  df['TIME_SINCE_LAST_SEARCH'] = df.groupby('USER_PSEUDO_ID')['EVENT_TIMESTAMP'].diff() / (1000*60*60)
  """

  return df

def extract_user_features(df_model):
    df = df_model.copy()

    df["user_search_frequency"] = df.groupby("USER_PSEUDO_ID").cumcount() + 1
    user_activity = df.groupby("USER_PSEUDO_ID").size()
    df["user_activity_level"] = df["USER_PSEUDO_ID"].map(user_activity)
    df["user_activity_score"] = np.log1p(df["user_activity_level"]) / np.log1p(user_activity.max())

    def search_diversity(group):
        return group.nunique() / (len(group) + 1)
    
    df["search_diversity"] = df.groupby("USER_PSEUDO_ID")["GEO_REGION"].transform(search_diversity)

    def calc_anywhere_ratio(user_id):
        user_rows = df[df["USER_PSEUDO_ID"] == user_id]
        anywhere_count = (user_rows["GEO_CITY_STANDARDIZED"] == "ANYWHERE").sum()
        total = len(user_rows)
        return anywhere_count / total if total > 0 else 0
    
    df["anywhere_ratio"] = df["USER_PSEUDO_ID"].map(lambda user_id: calc_anywhere_ratio(user_id))

    return df

# the recommendation engine
class LoadRecommendationEngine:
    def __init__(self, df_model, loads_df, top_k = 5):
        self.df_model = df_model
        self.loads_df = loads_df
        self.top_k = top_k
        self.vectorizer = None
        self.user_index = {user_index: index for index, user_index in enumerate(df_model["USER_PSEUDO_ID"].values)}
        self.similarity_matrix = None
        self._prepare_features()

    def _prepare_features(self):
        # initializes the tf-idf vectorizer & computes the similarity matrix

        combined_corpus = (
            list(self.df_model["route_history_weighted"].astype(str)) +
            list(self.loads_df["route_info"].astype(str))
        )

        self.vectorizer = TfidfVectorizer(max_features = 1000, stop_words="english", token_pattern = r"(?u)\b\w+\b")
        self.vectorizer.fit(combined_corpus)
        
        user_features = self.vectorizer.transform(self.df_model["route_history_weighted"])
        load_features = self.vectorizer.transform(self.loads_df["route_info"])

        self.similarity_matrix = cosine_similarity(user_features, load_features)

    # match the routes to users based on their search history of routes
    def _compute_route_similarity(self, user_idx):
        route_text = self.df_model.iloc[user_idx]["route_history_weighted"]
        route_text = route_text.upper()

        user_routes = set()
        user_tokens = set()

        for part in route_text.split():
            part = part.strip()
            if not part:
                continue

            if part == "ANYWHERE":
                user_routes.add("ANYWHERE")
            elif "," in part:
                city, state = part.split(",", 1)
                city = city.strip()
                state = state.strip()
                if len(city) > 1 and len(state) == 2:
                    user_routes.add(f"{city},{state}")
                    user_tokens.add(state)

        
        route_scores = np.zeros(len(self.loads_df))

        if len(user_routes) > 0:
            for load_index, load_route in enumerate(self.loads_df["route_info"]):
                load_parts = load_route.upper().split()
                best_match = 0.0
                
                for user_route in user_routes:
                    if user_route == "ANYWHERE":
                        continue
                    for load_part in load_parts:
                        if load_part == user_route:
                            best_match = max(best_match, 1.0)
                        elif "," in load_part and "," in user_route:
                            user_state = user_route.split(",")[1]
                            load_state = load_part.split(",")[1]
                            if user_state == load_state:
                                best_match = max(best_match, 0.5)
                route_scores[load_index] = best_match
            matching_loads = (route_scores > 0).sum()
            
        return route_scores
    
    # calculate the similarity based off of distances
    def _compute_geographic_similarity(self, user_idx):
        user_coordinates = self.df_model.iloc[user_idx]["user_coordinates"]

        distance_similarities = []

        for pickup_coordinates in self.loads_df["pickup_coord"]:
            distance = haversine(user_coordinates, pickup_coordinates)
            similarity = distance_to_similarity(distance)
            distance_similarities.append(similarity)

        return np.array(distance_similarities)
    
    # determprintine if there's not enough history for this user
    def _is_bad_user_history(self, user_idx):
        route_text = self.df_model.iloc[user_idx]["route_history_weighted"]

        if route_text == "ANYWHERE":
            return False
         
        tokens = route_text.split()

        if len(tokens) == 0:
            return True
        
        return False
    
    # calculate the hybrid (route & geo) similarity scores
    def get_hybrid_scores(self, user_idx, current_location=None):
        route_similarities = self._compute_route_similarity(user_idx)
        geo_similarities = self._compute_geographic_similarity(user_idx)
        current_location_similarities = self._compute_current_location_similarity(current_location)

        user_profile = infer_user_profile(self.df_model, user_idx)
        loads_personalized = calculate_personalized_load_quality(self.loads_df, user_profile)
        load_quality = loads_personalized["personalized_load_quality"].values

        route_text = self.df_model.iloc[user_idx]["route_history_weighted"]
        #is_anywhere = route_text == "ANYWHERE"
        anywhere_ratio = self.df_model.iloc[user_idx]["anywhere_ratio"]

        print(f"\n{'='*80}")
        print(f"USER ANALYSIS")
        print(f"{'='*80}")
        print(f"Route History: {route_text[:100]}..." if len(route_text) > 100 else f"Route History: {route_text}")
        print(f"Anywhere Ratio: {anywhere_ratio:.2%}")

        # weights:

        if current_location is not None:
            if self._is_bad_user_history(user_idx):
                w_route = 0.05
                w_current_loc = 0.8
                w_geo = 0.10
                w_quality = 0.10
                print(f"Strategy: SPARSE HISTORY + NATNAL")
                """
                # using as anywhere - base off of route history
                elif anywhere_ratio > 0.3:
                    w_route = 0.30
                    w_current_loc = 0.4
                    w_geo = 0.20
                    w_quality = 0.10
                    print(f"Strategy: ANYWHERE SEARCHER (ratio: {anywhere_ratio:.2f}) + NATNAL")
                """
            # there are specific routes - base more off of routes & geo
            else:
                w_route = 0.20
                w_current_loc = 0.6
                w_geo = 0.10
                w_quality = 0.10
                print(f"Strategy: SPECIFIC ROUTES")
            
            
            print(f"Weights - Route: {w_route:.2f}, Current Loc: {w_current_loc:.2f}, Geo: {w_geo:.2f}, Quality: {w_quality:.2f}")
            
            print(f"\nSimilarity Statistics:")
            print(f"  Route Sim:        min={route_similarities.min():.4f}, max={route_similarities.max():.4f}, mean={route_similarities.mean():.4f}, std={route_similarities.std():.4f}")
            print(f"  Current Loc Sim:  min={current_location_similarities.min():.4f}, max={current_location_similarities.max():.4f}, mean={current_location_similarities.mean():.4f}, std={current_location_similarities.std():.4f}")
            print(f"  Geo Sim:          min={geo_similarities.min():.4f}, max={geo_similarities.max():.4f}, mean={geo_similarities.mean():.4f}, std={geo_similarities.std():.4f}")
            print(f"  Load Quality:     min={load_quality.min():.4f}, max={load_quality.max():.4f}, mean={load_quality.mean():.4f}, std={load_quality.std():.4f}")


            activity_weight = 1.0

            hybrid_scores = (
                (w_route * route_similarities) +
                (w_current_loc * current_location_similarities) +
                (w_geo * geo_similarities) +
                (w_quality * load_quality)
            ) #* activity_weight

        else:
            # not enough search history - base off of geography
            if self._is_bad_user_history(user_idx):
                w_route = 0.20
                w_geo = 0.50
                w_quality = 0.30
                print(f"Strategy: SPARSE HISTORY")

            # using as anywhere - base off of route history
            elif anywhere_ratio > 0.3:
                w_route = 0.60
                w_geo = 0.10
                w_quality = 0.30
                print(f"Strategy: ANYWHERE SEARCHER (ratio: {anywhere_ratio:.2f})")
            
            # there are specific routes - base more off of routes & geo
            else:
                w_route = 0.70
                w_geo = 0.20
                w_quality = 0.10
                print(f"Strategy: SPECIFIC ROUTES")

            #print(f"Strategy: {strategy}")
            print(f"Weights - Route: {w_route:.2f}, Geo: {w_geo:.2f}, Quality: {w_quality:.2f}")
            
            print(f"\nSimilarity Statistics:")
            print(f"  Route Sim:    min={route_similarities.min():.4f}, max={route_similarities.max():.4f}, mean={route_similarities.mean():.4f}, std={route_similarities.std():.4f}")
            print(f"  Geo Sim:      min={geo_similarities.min():.4f}, max={geo_similarities.max():.4f}, mean={geo_similarities.mean():.4f}, std={geo_similarities.std():.4f}")
            print(f"  Load Quality: min={load_quality.min():.4f}, max={load_quality.max():.4f}, mean={load_quality.mean():.4f}, std={load_quality.std():.4f}")


            #activity_boost = self.df_model.iloc[user_idx]["user_activity_score"]
            #activity_weight = 1.0 + (activity_boost * 0.2)
            activity_weight = 1.0

            hybrid_scores = (
                (w_route * route_similarities) +
                (w_geo * geo_similarities) +
                (w_quality * load_quality)
            ) #* activity_weight

        return hybrid_scores
    
    # use if they have a NATNAL location
    def _compute_current_location_similarity(self, current_location):
        if current_location is None:
            return np.zeros(len(self.loads_df))
        
        print(f"\n{'='*80}")
        print(f"CURRENT LOCATION DISTANCE DEBUG")
        print(f"{'='*80}")
        print(f"User current location: {current_location}")
        
        distances=[]
        # get the distances to the loads
        distance_similarities = []
        for pickup_coordinates in self.loads_df["pickup_coord"]:
            distance = haversine(current_location, pickup_coordinates)
            similarity = distance_to_similarity(distance)
            distance_similarities.append(similarity)
            
            distances.append(distance)

        distances = np.array(distances)
        
        print(f"\nDistance Statistics:")
        print(f"  min={distances.min():.2f}mi, max={distances.max():.2f}mi, mean={distances.mean():.2f}mi")
        print(f"  Closest 5 loads:")
        
        closest_indices = np.argsort(distances)[:5]
        for i, idx in enumerate(closest_indices, 1):
            load = self.loads_df.iloc[idx]
            pickup = load['pickup']
            dist = distances[idx]
            sim = distance_similarities[idx]
            pickup_coord = self.loads_df.iloc[idx]['pickup_coord']
            print(f"    {i}. Load {load['id']}: {pickup['city']},{pickup['state']} at {pickup_coord}")
            print(f"       Distance: {dist:.2f}mi, Similarity: {sim:.4f}")

        return np.array(distance_similarities)
    
    # returns the recommended loads for the user, if they have a natnal or not
    def get_recommendations(self, user_id, current_location = None):
        if user_id not in self.user_index:
            return []
        
        user_idx = self.user_index[user_id]

        if current_location is not None:
            scores = self.get_hybrid_scores(user_idx, current_location=current_location)
        else:
            result = self.get_hybrid_scores(user_idx, current_location)
            scores = result
            #current_loc_sims = None

        top_indices = np.argsort(scores)[-self.top_k:][::-1]

        all_top_indices = np.argsort(scores)[-10:][::-1]

        route_similarities = self._compute_route_similarity(user_idx)
        geo_similarities = self._compute_geographic_similarity(user_idx)
        current_location_similarities = self._compute_current_location_similarity(current_location)
        
        user_profile = infer_user_profile(self.df_model, user_idx)
        loads_personalized = calculate_personalized_load_quality(self.loads_df, user_profile)
        load_quality = loads_personalized["personalized_load_quality"].values

        print(f"\n{'='*80}")
        print(f"TOP 10 LOADS RANKED")
        print(f"{'='*80}\n")
        
        all_top_indices = np.argsort(scores)[-10:][::-1]
        print(f"{'Rank':<5} {'Load':<6} {'Score':<10} {'Route':<10} {'CurrentLoc':<12} {'Geo':<10} {'Quality':<10} {'Route Match':<20}")
        print(f"{'-'*90}")

                
        for rank, load_idx in enumerate(all_top_indices, 1):
            load = self.loads_df.iloc[load_idx]
            score = scores[load_idx]
            route_comp = route_similarities[load_idx]
            geo_comp = geo_similarities[load_idx]
            quality_comp = load_quality[load_idx]
            current_loc_comp = current_location_similarities[load_idx]
            
            if route_comp > 0:
                if route_comp >= 1.0:
                    match_type = "Exact Route"
                else:
                    match_type = "Same State"
            else:
                match_type = "No Match"
            
            pickup = load['pickup']
            
            print(f"{rank:<5} {load['id']:<6} {score:<10.4f} {route_comp:<10.4f} {current_loc_comp:<12.4f} {geo_comp:<10.4f} {quality_comp:<10.4f} {match_type:<20}")
            print(f"       {pickup['city']},{pickup['state']}")
            print()

        recommendations = []
        for rank, load_idx in enumerate(top_indices, 1):
            load = self.loads_df.iloc[load_idx]
            score = scores[load_idx]
            recommendations.append({
                'rank': rank,
                'load_id': str(load['id']),
                'recommendation_score': float(score),
                'load_quality': float(load['load_quality']),
                'revenue_per_mile': float(load['revenue_per_mile']),
                'price': float(load['price']),
                'distance': float(load['distance']),
                'weight': float(load['weight']),
                'pickup': {
                    'city': load['pickup']['city'],
                    'state': load['pickup']['state'],
                    'date': load['pickup']['date'],
                    'time': load['pickup']['time']
                },
                'delivery': {
                    'city': load['delivery']['city'],
                    'state': load['delivery']['state'],
                    'date': load['delivery']['date'],
                    'time': load['delivery']['time']
                }
            })

        return recommendations
    
# global engine instance
engine = None

def initialize_engine(data_path = "click-stream(in).csv", loads_path = "mock_loads.json"):
    global engine
    
    # load the load data
    try: 
        loads_df = pd.read_json(loads_path)
    except FileNotFoundError:
        generate_mock_loads(save_path = loads_path)
        loads_df = pd.read_json(loads_path)

    # load the clickstream data
    try:
        print(f"Loading clickstream data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} clickstream events")

        # change the user ids to ints
        df['USER_PSEUDO_ID'] = df['USER_PSEUDO_ID'].astype(int)

        # normalize the df
        df = normalize_geo_to_event(df)

        df = df.sort_values(['USER_PSEUDO_ID', 'EVENT_TIMESTAMP']).reset_index(drop=True)

        # get users' prev searches
        df = get_prev_searches(df, n_features=3)

        # get their previous searches' origins - found to be an important feature from feature selection
        df['PREV_ORIGIN_1'] = df['PREV_ORIGIN_1'].mask(
            df['PREV_ORIGIN_1'].isin(['firebase', 'UNKNOWN']),
            df['GEO_CITY_STANDARDIZED']
        )

        df['PREV_ORIGIN_2'] = df['PREV_ORIGIN_2'].replace('firebase', "UNKNOWN")
        df['PREV_ORIGIN_3'] = df['PREV_ORIGIN_3'].replace('firebase', "UNKNOWN")

        user_searches = []
        for user_id, group in df.groupby("USER_PSEUDO_ID"):
            all_routes = set()

            geo_cities = group["GEO_CITY_STANDARDIZED"].dropna().unique()
            for geo_city in geo_cities:
                geo_city = str(geo_city).strip()

                if geo_city == "ANYWHERE":
                    all_routes.add("ANYWHERE")
                elif "," in geo_city:
                    city, state = geo_city.split(",", 1)
                    city = city.strip().upper()
                    state = state.strip().upper()

                    if len(city) > 1 and len(state) == 2:
                        all_routes.add(f"{city},{state}")

            user_activity_level = len(group)
            search_diversity = group["GEO_REGION"].nunique() / (len(group) + 1)
            anywhere_count = (group["GEO_CITY_STANDARDIZED"] == "ANYWHERE").sum()
            anywhere_ratio = anywhere_count / len(group) if len(group) > 0 else 0

            route_tokens = sorted(list(all_routes))
            route_history = " ".join(route_tokens) if route_tokens else "ANYWHERE"

            user_searches.append({
                "USER_PSEUDO_ID": user_id,
                "route_history_weighted": route_history,
                "user_activity_level": user_activity_level,
                "search_diversity": search_diversity,
                "anywhere_ratio": anywhere_ratio,
                "anywhere_count": anywhere_count,
                "GEO_CITY_STANDARDIZED": group["GEO_CITY_STANDARDIZED"].iloc[-1]
            })

        df_model = pd.DataFrame(user_searches)

    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return None
    except Exception as e:
        print(f"Error processing clickstream data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    loads_df = calculate_load_quality(loads_df)

    # get the cities & coordinates for the loads
    loads_df["pickup_city"] = loads_df["pickup"].apply(lambda x: f"{x['city']},{x['state']}")
    loads_df['pickup_coord'] = loads_df['pickup_city'].apply(get_latlon).apply(fix_coord)

    
    def build_route_info(pickup, delivery):
        pickup_route = f"{pickup['city'].upper()},{pickup['state'].upper()}"
        delivery_route = f"{delivery['city'].upper()},{delivery['state'].upper()}"
        return f"{pickup_route} {delivery_route}"
    
    # add the route info
    loads_df["route_info"] = loads_df.apply(lambda row: build_route_info(row["pickup"], row["delivery"]), axis = 1)

    df_model["user_search_frequency"] = 1
    user_activity_max = df_model["user_activity_level"].max()
    df_model["user_activity_score"] = np.log1p(df_model["user_activity_level"]) / np.log1p(user_activity_max)

    df_model["user_coordinates"] = df_model["GEO_CITY_STANDARDIZED"].apply(get_latlon).apply(fix_coord)

    engine = LoadRecommendationEngine(df_model, loads_df, top_k = 5)
    return engine


