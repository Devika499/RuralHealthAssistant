#!/usr/bin/env python3
"""
Database initialization script for Rural Healthcare AI Assistant
"""

import os
import sys
from sqlalchemy import create_engine, text
from database import DATABASE_URL, create_tables, Base

def create_database():
    """Create the database if it doesn't exist"""
    # Extract database name from URL
    db_name = DATABASE_URL.split('/')[-1]
    base_url = '/'.join(DATABASE_URL.split('/')[:-1])
    
    # Connect to PostgreSQL server (without specifying database)
    # Use autocommit mode for CREATE DATABASE
    engine = create_engine(base_url + '/postgres', isolation_level='AUTOCOMMIT')
    
    try:
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
            if not result.fetchone():
                # Create database
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                print(f"✅ Database '{db_name}' created successfully")
            else:
                print(f"✅ Database '{db_name}' already exists")
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False
    
    return True

def init_tables():
    """Initialize database tables"""
    try:
        create_tables()
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def main():
    """Main initialization function"""
    print("🚀 Initializing Rural Healthcare AI Assistant Database...")
    
    # Create database
    if not create_database():
        print("❌ Failed to create database. Please check your PostgreSQL connection.")
        sys.exit(1)
    
    # Create tables
    if not init_tables():
        print("❌ Failed to create tables.")
        sys.exit(1)
    
    print("🎉 Database initialization completed successfully!")
    print("\n📋 Next steps:")
    print("1. Make sure PostgreSQL is running on localhost:5432")
    print("2. Update DATABASE_URL in database.py if needed")
    print("3. Run the FastAPI server: python main.py")

if __name__ == "__main__":
    main() 