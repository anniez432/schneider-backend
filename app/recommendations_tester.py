import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class RecommendationTester:
        def __init__(self, df_original, df_model, loads_df, engine):
            self.df_original = df_original
            self.df_model = df_model
            self.loads_df = loads_df
            self.engine = engine
            self.results = []
        
        def print_summary(self) -> None:
            """Print overall accuracy summary across all tested users"""
            if not self.results:
                print("No results to summarize")
                return
            
            total_accuracy = 0
            total_tests = 0
            total_matches = 0
            total_no_matches = 0
            
            for result in self.results:
                match = result['match_analysis']
                total_accuracy += match['accuracy_score']
                total_tests += 1
                
                if result['distance_filter']:
                    total_matches += match['distance_within_range']
                elif result['current_location']:
                    total_matches += match['location_matches']
                else:
                    total_matches += match['history_matches']
                
                total_no_matches += match['no_matches']
            
            avg_accuracy = total_accuracy / total_tests if total_tests > 0 else 0
            total_recommendations = total_matches + total_no_matches
            
            print(f"\n{'='*80}")
            print(f"OVERALL RECOMMENDATION ENGINE SUMMARY")
            print(f"{'='*80}")
            print(f"\n  Tests Run: {total_tests}")
            print(f"  Average Accuracy: {avg_accuracy:.2%}")
            print(f"  Total Matches: {total_matches}/{total_recommendations}")
            print(f"  Total No Matches: {total_no_matches}/{total_recommendations}")
            print(f"\n{'='*80}\n")

        def get_user_history_stats(self, user_id: int) -> Dict:

            # get the info for this user
            user_data = self.df_original[self.df_original["USER_PSEUDO_ID"] == user_id]

            if len(user_data) == 0:
                return None
            
            # get the # times they searched/were in specific cities
            geo_city_counts = user_data['GEO_CITY_STANDARDIZED'].value_counts().to_dict()
            event_origin_counts = user_data['EVENT_ORIGIN'].value_counts().to_dict()
            event_destination_counts = user_data['EVENT_DESTINATION'].value_counts().to_dict()

            return {
            'user_id': user_id,
            'total_searches': len(user_data),
            'geo_city_counts': geo_city_counts,
            'event_origin_counts': event_origin_counts,
            'event_destination_counts': event_destination_counts,
            'top_geo_city': geo_city_counts.get(max(geo_city_counts, key=geo_city_counts.get), None) if geo_city_counts else None,
            'top_origin': event_origin_counts.get(max(event_origin_counts, key=event_origin_counts.get), None) if event_origin_counts else None,
            'top_destination': event_destination_counts.get(max(event_destination_counts, key=event_destination_counts.get), None) if event_destination_counts else None,
            }
        
        # get the pickup/dropoff location of specific load
        def get_location_from_load(self, load: pd.Series) -> Tuple[str, str]:
            pickup_location = f"{load['pickup']['city'].upper()},{load['pickup']['state'].upper()}"
            delivery_location = f"{load['delivery']['city'].upper()},{load['delivery']['state'].upper()}"
            return pickup_location, delivery_location

        # calculate how accurate the match is 
        def calculate_match_score(self, recommendations: List[Dict], user_history: Dict, current_location: Tuple = None, distance_range: int = None) -> Dict:
            if not recommendations or not user_history:
                return None

            geo_cities = user_history['geo_city_counts']
            top_locations = list(geo_cities.keys())[:5]

            top_states = set()
            for loc in top_locations:
                # if there's a comma in the location it has the state
                if ',' in loc:
                    top_states.add(loc.split(',')[1].strip())

            matches = {
                'location_matches': 0,
                'distance_within_range': 0,
                'history_matches': 0,
                'no_matches': 0,
                'matched_load_ids': [],
                'unmatched_load_ids': []
            }

            from recommendation import haversine

            # go through those top 5 recommendations
            for rec in recommendations:
                load = self.loads_df[self.loads_df['id'] == int(rec['load_id'])].iloc[0]
                pickup_loc, delivery_loc = self.get_location_from_load(load)

                # If current_location is specified, check if pickup is in that location
                if current_location is not None:
                    pickup_coord = load['pickup_coord']
                    distance = haversine(current_location, pickup_coord)
                    
                    # Check if within distance range if specified
                    if distance_range is not None:
                        if distance <= distance_range:
                            matches['distance_within_range'] += 1
                            matches['matched_load_ids'].append(rec['load_id'])
                        else:
                            matches['no_matches'] += 1
                            matches['unmatched_load_ids'].append(rec['load_id'])
                    else:
                        # No range specified, just check location match
                        matches['location_matches'] += 1
                        matches['matched_load_ids'].append(rec['load_id'])
                
                # If no current_location, check against user history
                else:
                    # check for exact location match (pickup or delivery)
                    if pickup_loc in top_locations or delivery_loc in top_locations:
                        matches['history_matches'] += 1
                        matches['matched_load_ids'].append(rec['load_id'])
                    # check for state match
                    else:
                        pickup_state = pickup_loc.split(',')[1].strip() if ',' in pickup_loc else None
                        delivery_state = delivery_loc.split(',')[1].strip() if ',' in delivery_loc else None
                        
                        if (pickup_state in top_states or delivery_state in top_states):
                            matches['history_matches'] += 1
                            matches['matched_load_ids'].append(rec['load_id'])
                        else:
                            matches['no_matches'] += 1
                            matches['unmatched_load_ids'].append(rec['load_id'])

            total = len(recommendations)
            
            if current_location is not None:
                if distance_range is not None:
                    accuracy = matches['distance_within_range'] / total if total > 0 else 0
                else:
                    accuracy = matches['location_matches'] / total if total > 0 else 0
            else:
                accuracy = matches['history_matches'] / total if total > 0 else 0

            return {
                'location_matches': matches['location_matches'],
                'distance_within_range': matches['distance_within_range'],
                'history_matches': matches['history_matches'],
                'no_matches': matches['no_matches'],
                'accuracy_score': accuracy,
                'matched_load_ids': matches['matched_load_ids'],
                'unmatched_load_ids': matches['unmatched_load_ids'],
                'top_user_locations': top_locations
            }
        
        # run the test for a user
        def test_user(self, user_id: int, current_location: Tuple = None, distance_range: int = None, desired_date: str = None, desired_time: str = None) -> Dict:
            user_history = self.get_user_history_stats(user_id)
            if not user_history:
                print("User not found")
                return None
            
            try:
                recommendations = self.engine.get_recommendations(
                    user_id, 
                    current_location=current_location,
                    distance_range=distance_range,
                    desired_date=desired_date,
                    desired_time=desired_time
                )
            except Exception as e:
                print(f"Error getting recommendations: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
            
            match_score = self.calculate_match_score(recommendations, user_history, current_location, distance_range)

            result = {
            'user_id': user_id,
            'user_history': user_history,
            'recommendations': recommendations,
            'match_analysis': match_score,
            'current_location': current_location,
            'distance_filter': {
                'distance_range': distance_range
            } if distance_range is not None and current_location else None,
            'datetime_filter': {
                'desired_date': desired_date,
                'desired_time': desired_time
            } if desired_date and desired_time else None
            }

            self.results.append(result)
            return result
        
        def print_report(self, result: Dict) -> None:
            if not result:
                return
            
            user_id = result['user_id']
            history = result['user_history']
            match = result['match_analysis']
            recs = result['recommendations']
            current_location = result['current_location']
            distance_filter = result['distance_filter']
            datetime_filter = result['datetime_filter']

            print(f"\n{'='*80}")
            print(f"TEST REPORT FOR USER {user_id}")
            print(f"{'='*80}\n")

            print(f"SEARCH HISTORY:")
            print(f"  Total Searches: {history['total_searches']}")
            print(f"  Top Geo City: {history['top_geo_city']}")
            print(f"  Top Origin: {history['top_origin']}")
            print(f"  Top Destination: {history['top_destination']}")
            
            print(f"\n  Top 5 Geo Locations:")
            for i, (loc, count) in enumerate(list(history['geo_city_counts'].items())[:5], 1):
                print(f"    {i}. {loc}: {count} searches")

            print(f"\n  Top 5 Origins:")
            for i, (origin, count) in enumerate(list(history['event_origin_counts'].items())[:5], 1):
                print(f"    {i}. {origin}: {count} searches")

            print(f"\n  Top 5 Destinations:")
            for i, (dest, count) in enumerate(list(history['event_destination_counts'].items())[:5], 1):
                print(f"    {i}. {dest}: {count} searches")

            if distance_filter:
                print(f"\nDISTANCE FILTER:")
                print(f"  Distance Range: 0 - {distance_filter['distance_range']} miles")

            if datetime_filter:
                print(f"\nDATETIME FILTER:")
                print(f"  Desired Date: {datetime_filter['desired_date']}")
                print(f"  Desired Time: {datetime_filter['desired_time']}")

            print(f"\n{'='*80}")
            print(f"RECOMMENDATION ACCURACY")
            print(f"{'='*80}\n")
            
            print(f"  Accuracy Score: {match['accuracy_score']:.2%}")
            
            if current_location is not None:
                if distance_filter is not None:
                    print(f"  Pickups Within Range: {match['distance_within_range']}/5")
                else:
                    print(f"  Location Matches: {match['location_matches']}/5")
            else:
                print(f"  Exact Location Matches: {match['history_matches']}/5")
                print(f"  State Matches: 0/5 (counted in accuracy)")
            
            print(f"  No Matches: {match['no_matches']}/5")
            
            print(f"\n  User's Top Locations (from history):")
            for i, loc in enumerate(match['top_user_locations'], 1):
                print(f"    {i}. {loc}")

            print(f"\nRECOMMENDED LOADS:")
            print(f"{'Rank':<5} {'Load ID':<8} {'Score':<10} {'Pickup':<25} {'Delivery':<25} {'Pickup Time':<15} {'Match':<15}")
            print(f"{'-'*110}")
            
            for i, rec in enumerate(recs, 1):
                load = self.loads_df[self.loads_df['id'] == int(rec['load_id'])].iloc[0]
                pickup = f"{load['pickup']['city']},{load['pickup']['state']}"
                delivery = f"{load['delivery']['city']},{load['delivery']['state']}"
                pickup_time = f"{load['pickup']['date']} {load['pickup']['time']}"
                
                if rec['load_id'] in match['matched_load_ids']:
                    match_status = "✓ Matched"
                else:
                    match_status = "✗ No Match"
                
                print(f"{i:<5} {rec['load_id']:<8} {rec['recommendation_score']:<10.4f} {pickup:<25} {delivery:<25} {pickup_time:<15} {match_status:<15}")

            print(f"\n{'='*80}\n")

if __name__ == "__main__":
    from recommendation import (
        initialize_engine, 
        normalize_geo_to_event,
        get_prev_searches,
        calculate_load_quality,
        get_latlon,
        fix_coord,
        generate_mock_loads
    )

    # get the arguments for RecommendationTester
    engine = initialize_engine(data_path="click-stream(in).csv", loads_path="mock_loads.json")

    df_original = pd.read_csv("click-stream(in).csv")
    df_original['USER_PSEUDO_ID'] = df_original['USER_PSEUDO_ID'].astype(int)
    df_original = normalize_geo_to_event(df_original)
    df_original = df_original.sort_values(['USER_PSEUDO_ID', 'EVENT_TIMESTAMP']).reset_index(drop=True)
    df_original = get_prev_searches(df_original, n_features=3)

    tester = RecommendationTester(df_original=df_original, df_model=engine.df_model, loads_df=engine.loads_df, engine=engine)
    
    # test the top 10 most appeared users 
    top_10_users = df_original['USER_PSEUDO_ID'].value_counts().head(10).index.tolist()

    # Define different test locations for variety
    test_locations = [
        {"name": "Denver, CO", "coords": (39.7392, -104.9903)},
        {"name": "New York, NY", "coords": (40.7128, -74.0060)},
        {"name": "Houston, TX", "coords": (29.7604, -95.3698)},
        {"name": "Chicago, IL", "coords": (41.8781, -87.6298)},
    ]

    # Example 1: Test without filters
    print("\n" + "="*80)
    print("TESTING WITHOUT FILTERS")
    print("="*80)
    for i, user_id in enumerate(top_10_users[:3], 1):
        result = tester.test_user(user_id=user_id)
        if result:
            tester.print_report(result)
        else:
            print("Failed to test")

    # Example 2: Test with datetime filter
    print("\n" + "="*80)
    print("TESTING WITH DATETIME FILTER")
    print("="*80)
    # Get a future date from the loads
    future_date = (datetime.now() + timedelta(days=5)).strftime("%b %d %Y")
    future_time = "10:00 AM"
    
    print(f"Filtering for pickups at or after: {future_date} {future_time}\n")
    
    for i, user_id in enumerate(top_10_users[:3], 1):
        result = tester.test_user(
            user_id=user_id,
            desired_date=future_date,
            desired_time=future_time
        )
        if result:
            tester.print_report(result)
        else:
            print("Failed to test")

    # Example 3: Test with distance filter
    print("\n" + "="*80)
    print("TESTING WITH DISTANCE FILTER")
    print("="*80)
    
    for loc_idx, location in enumerate(test_locations[:3], 1):
        print(f"\nLocation {loc_idx}: {location['name']}")
        print(f"Distance Range: 0 - 300 miles\n")
        
        user_id = top_10_users[loc_idx - 1]
        result = tester.test_user(
            user_id=user_id,
            current_location=location['coords'],
            distance_range=300
        )
        if result:
            tester.print_report(result)
        else:
            print("Failed to test")

    # Example 4: Test with both filters
    print("\n" + "="*80)
    print("TESTING WITH DISTANCE AND DATETIME FILTERS")
    print("="*80)
    future_date = (datetime.now() + timedelta(days=3)).strftime("%b %d %Y")
    future_time = "2:00 PM"
    
    for loc_idx, location in enumerate(test_locations[:3], 1):
        print(f"\nLocation {loc_idx}: {location['name']}")
        print(f"Distance Range: 0 - 250 miles")
        print(f"Filtering for pickups at or after: {future_date} {future_time}\n")
        
        user_id = top_10_users[loc_idx + 2] if loc_idx + 2 < len(top_10_users) else top_10_users[loc_idx - 1]
        result = tester.test_user(
            user_id=user_id,
            current_location=location['coords'],
            distance_range=250,
            desired_date=future_date,
            desired_time=future_time
        )
        if result:
            tester.print_report(result)
        else:
            print("Failed to test")
    
    # Print overall summary
    tester.print_summary()