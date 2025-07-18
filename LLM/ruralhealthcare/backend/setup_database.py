#!/usr/bin/env python3
"""
Quick database setup script for Rural Healthcare AI Assistant
"""

import os
import sys
from urllib.parse import quote_plus

def setup_database():
    """Interactive database setup"""
    print("🔧 Rural Healthcare AI Assistant - Database Setup")
    print("=" * 50)
    
    # Get database configuration
    print("\n📋 Database Configuration:")
    print("Default: postgresql://postgres:password@localhost:5432/rural_healthcare")
    
    # Ask for password
    password = input("\nEnter PostgreSQL password (or press Enter for 'password'): ").strip()
    if not password:
        password = "password"
    
    # URL-encode the password to handle special characters
    encoded_password = quote_plus(password)
    
    # Create DATABASE_URL
    database_url = f"postgresql://postgres:{encoded_password}@localhost:5432/rural_healthcare"
    
    # Update database.py
    try:
        with open("database.py", "r") as f:
            content = f.read()
        
        # Replace the password line
        import re
        new_content = re.sub(
            r'password = quote_plus\("[^"]*"\)',
            f'password = quote_plus("{password}")',
            content
        )
        
        with open("database.py", "w") as f:
            f.write(new_content)
        
        print("✅ Database configuration updated!")
        
    except Exception as e:
        print(f"❌ Error updating database configuration: {e}")
        return False
    
    # Try to initialize database
    print("\n🚀 Initializing database...")
    try:
        from init_db import main as init_main
        init_main()
        print("✅ Database setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        print("\n📋 Manual steps:")
        print("1. Make sure PostgreSQL is installed and running")
        print("2. Update the password in database.py if needed")
        print("3. Run: python init_db.py")
        return False

def main():
    """Main setup function"""
    if setup_database():
        print("\n🎉 Setup completed! You can now run:")
        print("python main.py")
    else:
        print("\n❌ Setup failed. Please check the manual steps above.")

if __name__ == "__main__":
    main() 