# Discord Online Tracker
Announces when someone goes online. My friend Alex typically lurks in our server, so it was funny to track how often he lurks. I am slowly changing function names and variables to reflect a generic tracker rather than Alex specific.

### Features
- Online Tracker
- Graphing of information
- Support for LLM model to talk like the person you're tracking
- JSON version or Firebase versions for storing online times

### .env File
Have an env file with the following information
```
DISCORD_BOT_TOKEN= # Discord Bot Token
API= # URL to LLM model, I self host mine
USER=1 # User you are tracking
CHANNEL= # Channel name 
AUDIO_COUNT= # Plays an audio file in vc
AUDIO_FILE= # Path to audio
CHANNEL_ID= # Channel ID for where tracking messages are sent
FIREBASE_SERVICE_ACCOUNT_KEY= # Path to your Firebase JSON
FIREBASE_DATABASE_URL= # URL to your Firebase database
```
