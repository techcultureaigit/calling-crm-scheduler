import asyncio
from app.core.logger import logger
from app.core.db import connect_to_mongo, close_mongo_connection, ensure_db_indexes
from app.services.scheduler_service import scheduler_loop

async def main():
    logger.info("Initializing standalone Scheduler Instance...")
    
    # Initialize DB connection
    connect_to_mongo()
    await ensure_db_indexes()
    
    try:
        # Run the scheduler loop indefinitely
        await scheduler_loop()
    except asyncio.CancelledError:
        logger.info("Scheduler loop cancelled.")
    except Exception as e:
        logger.error(f"Scheduler loop encountered a fatal error: {e}", exc_info=True)
    finally:
        # Cleanup
        close_mongo_connection()
        logger.info("Scheduler Instance shutdown gracefully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Scheduler Instance interrupted by user. Exiting.")
