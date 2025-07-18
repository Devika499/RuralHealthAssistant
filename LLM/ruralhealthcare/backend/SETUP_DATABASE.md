# PostgreSQL Database Setup Guide

This guide will help you set up PostgreSQL for the Rural Healthcare AI Assistant.

## Prerequisites

1. **PostgreSQL Installation**
   - Download from: https://www.postgresql.org/download/windows/
   - Or use a package manager like Chocolatey: `choco install postgresql`

2. **Python Dependencies** (already installed)
   - `sqlalchemy>=2.0.0`
   - `psycopg2-binary>=2.9.0`
   - `alembic>=1.12.0`

## Step 1: Install PostgreSQL

### Option A: Download from Official Website
1. Go to https://www.postgresql.org/download/windows/
2. Download the installer for Windows
3. Run the installer and follow the setup wizard
4. **Important**: Remember the password you set for the `postgres` user
5. Keep the default port (5432)

### Option B: Using Chocolatey (if you have it installed)
```bash
choco install postgresql
```

## Step 2: Verify PostgreSQL Installation

1. **Check if PostgreSQL is running**:
   ```bash
   # On Windows, check Services
   services.msc
   # Look for "postgresql-x64-15" (or similar)
   ```

2. **Test connection**:
   ```bash
   psql -U postgres -h localhost
   # Enter the password you set during installation
   ```

## Step 3: Configure Database Connection

1. **Update database URL** in `database.py` if needed:
   ```python
   DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/rural_healthcare"
   ```

2. **Default configuration**:
   - Host: `localhost`
   - Port: `5432`
   - Database: `rural_healthcare`
   - Username: `postgres`
   - Password: (the one you set during installation)

## Step 4: Initialize Database

1. **Run the initialization script**:
   ```bash
   python init_db.py
   ```

2. **Expected output**:
   ```
   🚀 Initializing Rural Healthcare AI Assistant Database...
   ✅ Database 'rural_healthcare' created successfully
   ✅ Database tables created successfully
   🎉 Database initialization completed successfully!
   ```

## Step 5: Verify Database Setup

1. **Connect to the database**:
   ```bash
   psql -U postgres -d rural_healthcare
   ```

2. **Check tables**:
   ```sql
   \dt
   -- Should show: users, chat_messages, medical_records
   ```

3. **Exit psql**:
   ```sql
   \q
   ```

## Troubleshooting

### Common Issues:

1. **"Connection refused"**:
   - Make sure PostgreSQL service is running
   - Check if port 5432 is not blocked by firewall

2. **"Authentication failed"**:
   - Verify the password in DATABASE_URL
   - Try connecting with psql to test credentials

3. **"Database does not exist"**:
   - Run `python init_db.py` to create the database

4. **"Permission denied"**:
   - Make sure the postgres user has proper permissions
   - Try running as administrator if needed

### Environment Variables

You can also set the database URL as an environment variable:
```bash
set DATABASE_URL=postgresql://postgres:password@localhost:5432/rural_healthcare
```

## Next Steps

After successful database setup:

1. **Start the backend**:
   ```bash
   python main.py
   ```

2. **Test registration** through the frontend
3. **Verify data persistence** by checking the database

## Database Schema

The application creates three main tables:

1. **users**: User accounts and profiles
2. **chat_messages**: Chat history with AI responses
3. **medical_records**: Medical history records

All data will now persist between server restarts! 