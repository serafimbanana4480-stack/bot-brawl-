from api_server import get_bot
import sys
import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Initializing bot...")
bot = get_bot()

# Forcing auto-tuning off for testing to isolate core
if hasattr(bot, 'auto_tuning_enabled'):
    bot.auto_tuning_enabled = False

print("Setting up bot...")
success = bot.setup()
print(f"Bot setup status: {success}")

if success:
    print("Starting bot...")
    bot.start()
    
    print("Bot is running. Waiting 5 seconds to verify stability...")
    time.sleep(5)
    
    print("Bot Status:")
    print(bot.get_status())
    
    print("Stopping bot...")
    bot.stop()
    print("Test completed successfully.")
    sys.exit(0)
else:
    print("Setup failed!")
    sys.exit(1)
