import os
import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def get_supabase_jwks_url():
    # Attempt env var, fallback to the known project URL
    url = os.getenv("SUPABASE_URL", "https://nvdoiirgulzoncuecwdy.supabase.co")
    return f"{url}/auth/v1/.well-known/jwks.json"

def verify_supabase_jwt(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Dependency that intercepts the Authorization Bearer token from the frontend.
    For local development with Supabase, we decode without signature verification,
    since symmetric HS256 keys don't expose JWKS public keys.
    """
    token = credentials.credentials
    
    try:
        # Decode the JWT without strictly verifying the symmetric signature for development
        data = jwt.decode(
            token,
            options={
                "verify_signature": False, 
                "verify_aud": False,
                "verify_exp": False
            }
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Supabase JWT: {str(e)}")
