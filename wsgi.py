#!/usr/bin/env python3
"""
Minimal WSGI entry point for Render deployment
"""

import os
import sys

# Set environment variables
os.environ['FLASK_ENV'] = 'production'

print("🔧 WSGI: Starting application import...")
print(f"🔧 WSGI: Python version: {sys.version}")
print(f"🔧 WSGI: PORT environment variable: {os.environ.get('PORT', 'not set')}")
print(f"🔧 WSGI: FLASK_ENV: {os.environ.get('FLASK_ENV', 'not set')}")

try:
    from app import app
    print("✅ WSGI: App imported successfully")
    
    # Configure for production
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    
    print(f"✅ WSGI: App configured for production")
    print(f"🌐 WSGI: App debug mode: {app.debug}")
    
    # This is what gunicorn will use
    application = app
    
    print("✅ WSGI: Application ready for gunicorn")
    
except Exception as e:
    print(f"❌ WSGI: Failed to import app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
