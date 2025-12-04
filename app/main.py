from fastapi import FastAPI, HTTPException
from recommendation import initialize_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Load Recommendation API", version="1.0.0")

# Initialize the recommendation engine on startup
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        logger.info("Initializing recommendation engine...")
        engine = initialize_engine(
            data_path="click-stream(in).csv",
            loads_path="mock_loads.json"
        )
        if engine is None:
            logger.error("Failed to initialize recommendation engine")
        else:
            logger.info("Recommendation engine initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing engine: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "engine_initialized": engine is not None}

@app.get("/recommend/{user_id}")
def recommend(
    user_id: int, 
    current_lat: float = None, 
    current_lon: float = None, 
    distance_range: int = None,
    limit: int = 5,
    page: int = 1,
    desired_date: str = None,
    desired_time: str = None
):
    """Get load recommendations for a user
    
    Args:
        user_id: User ID
        current_lat: User's current latitude (optional)
        current_lon: User's current longitude (optional)
        distance_range: Maximum distance in miles for pickup locations from current location (optional)
                        Only applies if current_lat and current_lon are provided
        limit: Number of recommendations to return (default: 5)
        page: Page number (default: 1)
        desired_date: Filter loads for pickups at or after this date (optional)
                      Format: "Dec 3 2025" or "12/3/2025"
        desired_time: Filter loads for pickups at or after this time (optional)
                      Format: "2:30 PM" (12-hour with AM/PM)
    
    Example:
        /recommend/1 (no current location)
        /recommend/1?current_lat=39.0997&current_lon=-94.5786 (Kansas City)
        /recommend/1?current_lat=39.0997&current_lon=-94.5786&max_distance=300
        /recommend/1?limit=10&page=2
        /recommend/1?desired_date=12/15/2025&desired_time=2:30 PM
        /recommend/1?current_lat=39.0997&current_lon=-94.5786&max_distance=300&desired_date=Dec 15 2025&desired_time=3:45 PM&limit=10&page=2
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    
    try:
        current_location = None
        if current_lat is not None and current_lon is not None:
            current_location = (current_lat, current_lon)
            logger.info(f"Recommendation for user {user_id} from location ({current_lat}, {current_lon}) - Page {page}, Limit {limit}")
            if distance_range is not None:
                logger.info(f"Distance filter: max {distance_range} miles")
        else:
            logger.info(f"Recommendation for user {user_id} (no current location) - Page {page}, Limit {limit}")
        
        if desired_date and desired_time:
            logger.info(f"Filtering for pickups at or after: {desired_date} {desired_time}")

        recommendations = engine.get_recommendations(user_id, current_location, distance_range = distance_range, limit=limit, page=page, desired_date = desired_date, desired_time = desired_time)
        
        if not recommendations:
            return {
                "user_id": user_id,
                "current_location": current_location,
                "distance_filter":{
                    "distance_range": distance_range
                } if distance_range is not None and current_location else None,
                "datetime_filter": {
                    "desired_date": desired_date,
                    "desired_time": desired_time
                } if desired_date and desired_time else None,
                "recommendations": [],
                "message": "No recommendations available for this user"
            }
        
        return {
            "user_id": user_id,
            "current_location": current_location,
            "distance_filter": {
                "max_distance": distance_range
            } if distance_range is not None and current_location else None,
            "datetime_filter": {
                "desired_date": desired_date,
                "desired_time": desired_time
            } if desired_date and desired_time else None,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Load Recommendation API",
        "endpoints": {
            "health": "/health",
            "recommendations": "/recommend/{user_id}",
            "docs": "/docs"
        }
    }