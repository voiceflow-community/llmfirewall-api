from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llamafirewall import LlamaFirewall, UserMessage, Role, ScannerType, ScanResult
from typing import Optional, Dict, List
import asyncio
import os
import json

app = FastAPI(
    title="LlamaFirewall API",
    description="API for scanning user messages using LlamaFirewall",
    version="1.0.0"
)

def parse_scanners_config() -> Dict[Role, List[ScannerType]]:
    """
    Parse scanner configuration from environment variable.
    Default configuration if not set:
    {
        "USER": ["PROMPT_GUARD"]
    }
    """
    default_config = {
        Role.USER: [ScannerType.PROMPT_GUARD]
    }
    
    config_str = os.getenv("LLAMAFIREWALL_SCANNERS", "{}")
    try:
        # Parse the JSON configuration
        config_dict = json.loads(config_str)
        
        # Convert string keys to Role enum and string values to ScannerType enum
        scanners = {}
        for role_str, scanner_list in config_dict.items():
            try:
                role = Role[role_str]
                scanners[role] = [ScannerType[scanner] for scanner in scanner_list]
            except KeyError as e:
                raise ValueError(f"Invalid role or scanner type: {e}")
        
        return scanners if scanners else default_config
    except json.JSONDecodeError:
        print("Warning: Invalid JSON in LLAMAFIREWALL_SCANNERS, using default configuration")
        return default_config

# Initialize LlamaFirewall with configurable scanners
llamafirewall = LlamaFirewall(
    scanners=parse_scanners_config()
)

class ScanRequest(BaseModel):
    content: str

#class ScanResponse(BaseModel):
#    is_safe: bool
#    risk_score: float
#    details: dict

@app.post("/scan", response_model=ScanResult)
async def scan_message(request: ScanRequest):
    """
    Scan a user message for potential security risks.
    
    Args:
        request: ScanRequest containing the message content
        
    Returns:
        ScanResult containing the scan results
        
    Raises:
        HTTPException: If there's an error during scanning
    """
    try:
        message = UserMessage(
            content=request.content
        )
        
        # Run the scan in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: llamafirewall.scan(message))
        return result
        
        #return ScanResponse(
        #    is_safe=result.is_safe,
        #    risk_score=result.risk_score,
        #    details=result.details
        #)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error scanning message: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy"}

@app.get("/config")
async def get_config():
    """Get the current scanner configuration."""
    return {
        "scanners": {
            role.name: [scanner.name for scanner in scanners]
            for role, scanners in llamafirewall.scanners.items()
        }
    } 