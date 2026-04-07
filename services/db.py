"""Supabase connection helper.

Exposes an **async-friendly** Supabase client built from environment
variables.  Every service file should import ``supabase`` from here.
"""

import os
import logging

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
logger = logging.getLogger(__name__)

SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = (
    os.environ.get("SUPABASE_SERVICE_KEY", "")
    or os.environ.get("SUPABASE_KEY", "")
)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "FATAL: SUPABASE_URL and one of SUPABASE_SERVICE_KEY/SUPABASE_KEY must be set in the environment."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
logger.info("Supabase client initialised for %s", SUPABASE_URL)