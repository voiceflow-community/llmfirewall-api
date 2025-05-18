from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llamafirewall import LlamaFirewall, UserMessage, Role, ScannerType, ScanResult
from typing import Optional, Dict, List
import asyncio
import os
import json
from openai import OpenAI

app = FastAPI(
    title="LlamaFirewall API",
    description="API for scanning user messages using LlamaFirewall and OpenAI moderation",
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
    global moderation
    moderation = False  # Reset moderation flag
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
                # Handle MODERATION scanner type separately
                scanners[role] = []
                for scanner in scanner_list:
                    if scanner == "MODERATION":
                        # Check if OpenAI API key is configured
                        if not os.getenv("OPENAI_API_KEY"):
                            raise ValueError("OPENAI_API_KEY environment variable is required when using MODERATION scanner")
                        moderation = True
                    else:
                        scanners[role].append(ScannerType[scanner])
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

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

class ModerationResult(BaseModel):
    """Model for a single moderation result."""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]

class OpenAIModerationResponse(BaseModel):
    """Model for OpenAI moderation response."""
    id: str
    model: str
    results: List[ModerationResult]

class ScanRequest(BaseModel):
    content: str

class ScanResponse(BaseModel):
    """Unified response model for both scan types."""
    is_safe: bool
    risk_score: Optional[float] = None
    details: Optional[dict] = None
    moderation_results: Optional[OpenAIModerationResponse] = None
    scan_type: str

@app.post("/scan", response_model=ScanResponse)
async def scan_message(request: ScanRequest):
    """
    Scan a user message for potential security risks using LlamaFirewall and optionally OpenAI moderation.
    LlamaFirewall is always used, and OpenAI moderation is added when MODERATION scanner is enabled.
    
    Args:
        request: ScanRequest containing the message content
        
    Returns:
        ScanResponse containing the scan results and metadata
        
    Raises:
        HTTPException: If there's an error during scanning or if OpenAI API key is missing when MODERATION is enabled
    """
    try:
        # Always use LlamaFirewall first
        message = UserMessage(content=request.content)
        loop = asyncio.get_event_loop()

        llama_result = await loop.run_in_executor(None, lambda: llamafirewall.scan(message))

        # Initialize response with LlamaFirewall results
        response = ScanResponse(
            is_safe=True if llama_result.decision == "allow" else False,
            risk_score=llama_result.score,
            details={"reason": llama_result.reason},
            scan_type="llamafirewall"
        )
        
        # Add OpenAI moderation if enabled
        if moderation:
            if not os.getenv("OPENAI_API_KEY"):
                raise HTTPException(
                    status_code=500,
                    detail="OPENAI_API_KEY environment variable is required when using MODERATION scanner"
                )

            try:
                openai_response = client.moderations.create(
                    model="omni-moderation-latest",
                    input=request.content
                )

                # Convert OpenAI response to our model
                moderation_response = OpenAIModerationResponse(
                    id=openai_response.id,
                    model=openai_response.model,
                    results=[
                        ModerationResult(
                            flagged=result.flagged,
                            categories=result.categories.model_dump(),
                            category_scores=result.category_scores.model_dump()
                        )
                        for result in openai_response.results
                    ]
                )

                # Update response with moderation results
                response.moderation_results = moderation_response
                response.scan_type = "llamafirewall+openai_moderation"

                # Consider message unsafe if either LlamaFirewall or OpenAI flags it
                if any(result.flagged for result in moderation_response.results):
                    response.is_safe = False
                    # Add flagged categories to details
                    if response.details is None:
                        response.details = {}
                    response.details["flagged_categories"] = {
                        category: score
                        for result in moderation_response.results
                        for category, score in result.category_scores.items()
                        if score > 0.5
                    }
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error during OpenAI moderation: {str(e)}"
                )
        
        return response

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
    config = {
        "scanners": {
            role.name: [scanner.name for scanner in scanners]
            for role, scanners in llamafirewall.scanners.items()
        }
    }

    # Add MODERATION to the list if enabled
    if moderation:
        for role in config["scanners"]:
            if "MODERATION" not in config["scanners"][role]:
                config["scanners"][role].append("MODERATION")

    return config