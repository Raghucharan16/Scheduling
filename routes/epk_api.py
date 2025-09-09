from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import date
import database_service
import interpolate_epk_data

router = APIRouter()

@router.get("/epk-data")
async def generate_epk_data(
    sector: str = Query(..., description="Sector name (e.g., 'B-T', 'H-V', 'VSKP-VJY')"),
    from_date: date = Query(..., description="Start date for journey range"),
    to_date: date = Query(..., description="End date for journey range"),
    data_source: str = Query(..., description="Data source ('fb' for FreshBus or 'neugo' for Neugo)")
):
    """
    Generate EPK data for a given sector and date range.
    
    Args:
        sector: The route sector (e.g., 'B-T', 'H-V', 'VSKP-VJY')
        from_date: Start date for the journey range
        to_date: End date for the journey range  
        data_source: Data source ('fb' for FreshBus or 'neugo' for Neugo)
    
    Returns:
        Status message indicating successful data generation
    """
    try:
        # Convert dates to string format expected by database service
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')
        
        # Query raw data from database
        raw_data = database_service.get_epk_data(
            sector=sector,
            from_date=from_date_str,
            to_date=to_date_str,
            data_source=data_source
        )
        
        if raw_data is None or raw_data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for sector '{sector}' in date range {from_date_str} to {to_date_str}"
            )
        
        # Interpolate the EPK data
        interpolated_data = interpolate_epk_data.process_epk_data(raw_data, sector)
        
        return {
            "status": "success",
            "message": f"EPK data generated successfully for sector '{sector}'",
            "data_source": data_source,
            "date_range": {
                "from": from_date_str,
                "to": to_date_str
            },
            "records_count": len(interpolated_data),
            "sector": sector
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
