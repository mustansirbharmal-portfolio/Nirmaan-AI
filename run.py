#!/usr/bin/env python3
"""
Nirmaan AI Communication Scorer - Production Entry Point
WSGI application entry point for production deployment (Render, Heroku, etc.)
"""

import os
import sys
from app import app

# Production configuration
if os.getenv('FLASK_ENV') == 'production':
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
else:
    app.config['DEBUG'] = True

# For WSGI servers (gunicorn, etc.)
application = app

if __name__ == '__main__':
    # Development server configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    # Check if all required environment variables are set
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_DEPLOYMENT_NAME'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("⚠️  Warning: Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nThe application may not function properly without these variables.")
        if debug:
            print("Please check your .env file.\n")
        else:
            print("Please configure environment variables in your deployment platform.\n")
    
    if debug:
        print("🚀 Starting Nirmaan AI Communication Scorer (Development)...")
        print("📊 Features:")
        print("   - Rule-based scoring (keywords, word count)")
        print("   - NLP semantic analysis")
        print("   - Azure OpenAI integration")
        print("   - Comprehensive rubric evaluation")
        print("   - Real-time feedback")
        print(f"\n🌐 Access the application at: http://localhost:{port}")
        print("📖 API documentation available in README.md")
        print("\n" + "="*50)
    else:
        print(f"🚀 Starting Nirmaan AI Communication Scorer (Production) on port {port}...")
    
    try:
        app.run(
            debug=debug,
            host='0.0.0.0',
            port=port,
            use_reloader=debug
        )
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)
