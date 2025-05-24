from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from llamafirewall import LlamaFirewall, UserMessage, Role, ScannerType, ScanDecision
from typing import Optional, Dict, List, Pattern
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import os
import json
import logging
import logging.handlers
from datetime import datetime
import re

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = f"llmfirewall_api_{datetime.now().strftime('%Y%m%d')}.log"

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure root logger
logger = logging.getLogger("llmfirewall_api")
logger.setLevel(getattr(logging, LOG_LEVEL))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(console_handler)

# File handler with rotation
file_handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join("logs", LOG_FILE),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(file_handler)

# Get thread pool configuration from environment
THREAD_POOL_WORKERS = int(os.getenv("THREAD_POOL_WORKERS", "4"))  # Default to 4 workers if not specified
logger.info(f"Initializing thread pool with {THREAD_POOL_WORKERS} workers")

# Create thread pool executor at startup
thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)

app = FastAPI(
    title="LLM Firewall API",
    description="API for scanning user messages using LlamaFirewall and OpenAI moderation",
    version="1.1.0"
)

# Pre-compile regex patterns
INJECTION_PATTERNS: List[Pattern] = [
    re.compile(pattern, re.IGNORECASE) for pattern in [
        r'<\?php',
        r'<script',
        r'javascript:',
        r'data:',
        r'vbscript:',
        r'on\w+=',
        r'exec\s*\(',
        r'eval\s*\(',
        r'system\s*\(',
        r'base64_decode\s*\(',
        r'from\s+import\s+',
        r'__import__\s*\(',
    ]
]

def sanitize_log_data(data: dict) -> dict:
    """Sanitize sensitive data for logging."""
    if not data:
        return data
    sanitized = data.copy()
    sensitive_fields = {'content', 'api_key', 'token', 'error'}
    for field in sensitive_fields:
        if field in sanitized:
            sanitized[field] = '[REDACTED]'
    return sanitized

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
    logger.debug(f"Parsing scanner configuration: {config_str}")

    if "MODERATION" in config_str:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when using MODERATION scanner")
        moderation = True
        logger.info("OpenAI moderation enabled")

    # Parse the JSON configuration
    try:
        config_dict = json.loads(config_str)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in scanner configuration", extra=sanitize_log_data({"error": str(e)}))
        raise ValueError("Invalid JSON format in scanner configuration") from e

    # Validate configuration structure
    if not isinstance(config_dict, dict):
        raise ValueError("Invalid scanner configuration: must be a JSON object")

    # Convert string keys to Role enum and string values to ScannerType enum
    scanners = {}
    for role_str, scanner_list in config_dict.items():
        if not isinstance(scanner_list, list):
            raise ValueError(f"Invalid scanner list for role {role_str}")

        try:
            role = Role[role_str]
            scanners[role] = []
            for scanner in scanner_list:
                if not isinstance(scanner, str):
                    raise ValueError(f"Invalid scanner type format: {scanner}")
                if scanner != "MODERATION":
                    scanners[role].append(ScannerType[scanner])
        except KeyError as e:
            raise ValueError(f"Invalid scanner configuration: {e}")

    logger.info("Scanner configuration loaded successfully")
    return scanners if scanners else default_config

# Initialize application with proper error handling
try:
    # Cache scanner configuration at startup
    logger.info("Initializing scanner configuration")
    SCANNER_CONFIG = parse_scanners_config()

    # Initialize LlamaFirewall with cached config
    logger.info("Initializing LlamaFirewall")
    llamafirewall = LlamaFirewall(scanners=SCANNER_CONFIG)

    # Initialize OpenAI client if moderation is enabled
    if moderation:
        logger.info("Initializing OpenAI client")
        async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    else:
        async_client = None

except ValueError as e:
    logger.error(f"Failed to initialize application: {e}", extra=sanitize_log_data({"error": str(e)}))
    exit(1)
except Exception as e:
    logger.error(f"Unexpected error during initialization: {e}", extra=sanitize_log_data({"error": str(e)}))
    exit(1)

class ModerationResult(BaseModel, frozen=True):
    """Model for a single moderation result."""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]

class OpenAIModerationResponse(BaseModel, frozen=True):
    """Model for OpenAI moderation response."""
    id: str
    model: str
    results: List[ModerationResult]

class ScanRequest(BaseModel, frozen=True):
    """Request model for scanning messages."""
    content: str = Field(min_length=1, max_length=10000)  # Constrain content length between 1 and 10000 characters

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content for potential security issues."""
        # Check for injection patterns
        for pattern in INJECTION_PATTERNS:
            if pattern.search(v):
                raise ValueError("Content contains potentially unsafe patterns")

        # Check for excessive whitespace (potential DoS)
        if len(v.strip()) == 0:
            raise ValueError("Content cannot be empty or whitespace only")

        # Check for excessive repeated characters (potential DoS)
        if any(c * 100 in v for c in set(v)):
            raise ValueError("Content contains excessive repeated characters")

        return v

class ScanResponse(BaseModel, frozen=True):
    """Unified response model for both scan types."""
    is_safe: bool
    risk_score: Optional[float] = None
    details: Optional[dict] = None
    moderation_results: Optional[OpenAIModerationResponse] = None
    scan_type: str

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def perform_openai_moderation(content: str) -> OpenAIModerationResponse:
    """Perform OpenAI moderation with retry logic."""
    try:
        logger.debug("Performing OpenAI moderation", extra=sanitize_log_data({"content": content}))
        response = await async_client.moderations.create(
            model="omni-moderation-latest",
            input=content
        )

        result = OpenAIModerationResponse(
            id=response.id,
            model=response.model,
            results=[
                ModerationResult(
                    flagged=result.flagged,
                    categories=result.categories.model_dump(),
                    category_scores=result.category_scores.model_dump()
                )
                for result in response.results
            ]
        )

        if any(r.flagged for r in result.results):
            logger.warning("Content flagged by OpenAI moderation", extra=sanitize_log_data({
                "flagged": True,
                "categories": {k: v for r in result.results for k, v in r.categories.items() if v}
            }))
        else:
            logger.debug("Content passed OpenAI moderation")

        return result
    except Exception as e:
        logger.error("Error during content moderation", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
        raise HTTPException(
            status_code=500,
            detail="Error during content moderation"
        )

@app.post("/scan", response_model=ScanResponse)
async def scan_message(request: ScanRequest):
    """
    Scan a user message for potential security risks using LlamaFirewall and optionally OpenAI moderation.
    LlamaFirewall is always used, and OpenAI moderation is added when moderation=True.
    
    Args:
        request: ScanRequest containing the message content
        
    Returns:
        ScanResponse containing the scan results and metadata
        
    Raises:
        HTTPException: If there's an error during scanning
    """
    logger.info("Received scan request", extra=sanitize_log_data({"content": request.content}))
    try:
        # Always use LlamaFirewall first
        message = UserMessage(content=request.content)
        logger.debug("Starting LlamaFirewall scan")

        try:
            # Use thread pool for LlamaFirewall scan
            llama_result = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: llamafirewall.scan(message)
            )
        except Exception as e:
            logger.error("LlamaFirewall scan failed", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable"
            )

        # Log sanitized result
        logger.info("LlamaFirewall scan completed", extra=sanitize_log_data({
            "decision": llama_result.decision,
            "score": llama_result.score,
            "reason": llama_result.reason
        }))

        response = ScanResponse(
            is_safe=True if llama_result.decision == ScanDecision.ALLOW else False,
            risk_score=llama_result.score,
            details={"reason": llama_result.reason},
            scan_type="llamafirewall"
        )

        if moderation:
            try:
                logger.debug("Starting OpenAI moderation")
                # Perform OpenAI moderation asynchronously
                moderation_response = await perform_openai_moderation(request.content)

                # Update response with moderation results
                response = ScanResponse(
                    is_safe=response.is_safe and not any(r.flagged for r in moderation_response.results),
                    risk_score=response.risk_score,
                    details={
                        "reason": response.details.get("reason", ""),
                        "flagged_categories": {
                            category: score
                            for result in moderation_response.results
                            for category, score in result.category_scores.items()
                            if score > 0.5
                        } if any(r.flagged for r in moderation_response.results) else None
                    },
                    moderation_results=moderation_response,
                    scan_type="llamafirewall+openai_moderation"
                )
                logger.info("Moderation scan completed", extra=sanitize_log_data({
                    "is_safe": response.is_safe,
                    "flagged_categories": response.details.get("flagged_categories")
                }))
            except HTTPException:
                raise
            except Exception as e:
                logger.error("OpenAI moderation failed", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable"
                )

        return response

    except HTTPException:
        raise
    except ValueError as e:
        # Handle validation errors
        logger.error("Validation error", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
        raise HTTPException(
            status_code=400,
            detail="Invalid request format"
        )
    except Exception as e:
        logger.error("Unexpected error", exc_info=True, extra=sanitize_log_data({"error": str(e)}))
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    logger.debug("Health check requested")
    return {"status": "healthy"}

@app.get("/config")
async def get_config():
    """Get the current scanner configuration."""
    logger.debug("Config requested")
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

    logger.debug(f"Returning config: {config}")
    return config

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")
    thread_pool.shutdown(wait=True)
    logger.info("Thread pool shutdown complete")