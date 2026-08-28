# -*- coding: utf-8 -*-
"""Production entrypoint for Cloud Deployment (Render, Railway, Heroku, Fly.io, etc.)."""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8766))
    # Run the integrated CS + Shadow + Tree navigation app
    uvicorn.run("integrated.app:app", host="0.0.0.0", port=port, reload=False)
