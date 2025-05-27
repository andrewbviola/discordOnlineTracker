import discord
import pytz
from discord.ext import commands, tasks
from datetime import datetime, timedelta, time
from collections import defaultdict
import json # Keep for potential API interactions, but not file storage
import asyncio
import os
import matplotlib.pyplot as plt
import io
import requests
from dotenv import load_dotenv
import firebase_admin # Import Firebase
from firebase_admin import credentials, db # Import credentials and db
import re

load_dotenv()

# --- Environment Variables ---
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
# DATA_FILE = "online_times.json" # Removed
# NO_MESSAGE_FILE = "no_message_data.json" # Removed
# AUDIO_COUNT_FILE = os.getenv("AUDIO_COUNT") # Removed
AUDIO_FILE = os.getenv("AUDIO_FILE")
FIREBASE_KEY_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
FIREBASE_DB_URL = os.getenv("FIREBASE_DATABASE_URL")

LURKIN = "<:lurkin:1275247450285932565>"
LURKER = "<:lurker:1257490595266560071>"

TRIPLE_GIF="https://tenor.com/view/rodrick-camera-point-rodrick-heffley-diary-of-a-wimpy-kid-gif-13001614153341977897"

CHANNEL = os.getenv("CHANNEL")
USER_ID = int(os.getenv("USER_ID")) # Ensure USER_ID is an integer
API_URL = os.getenv("API")
CHANNEL_ID=int(os.getenv("CHANNEL_ID")) # Ensure CHANNEL_ID is an integer

CHAT_KEY = os.getenv("CHAT_KEY")
ALEX_KEY = os.getenv("ALEX_KEY")

# --- Firebase Initialization ---
try:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': FIREBASE_DB_URL
    })
    print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")
    # Consider exiting or handling this error appropriately
    exit()

# --- Bot Setup ---
intents = discord.Intents.all()
intents.presences = True
intents.members = True
triggered_messages = set()

bot = commands.Bot(command_prefix="!", intents=intents)
eastern = pytz.timezone("America/New_York")

# --- In-memory state (will be loaded from Firebase) ---
# Use defaultdicts initially, but they will be replaced by data from Firebase
user_online_times = defaultdict(lambda: defaultdict(list))
user_status = defaultdict(
    lambda: {"online": False, "message_sent": False, "online_count": 0}
)
# regular_transitions is ephemeral daily data, might not need Firebase persistence
regular_transitions = defaultdict(lambda: {"online": 0, "offline": 0})
no_message_data = defaultdict(int)
audio_counts = {"count": 0} # Simple dict for audio count

# --- Firebase Data Loading ---
def load_data_from_firebase():
    global user_online_times, no_message_data, audio_counts
    print("Loading data from Firebase...")
    try:
        # Load online times
        online_times_ref = db.reference('/online_times')
        loaded_online_times = online_times_ref.get()
        if loaded_online_times:
            # Convert Firebase data back to nested defaultdict structure if needed
            # Firebase returns dicts, which might be sufficient
            user_online_times.update(loaded_online_times)
            print(f"Loaded {len(user_online_times)} days of online times.")
        else:
            print("No existing online times found in Firebase.")

        # Load no message data
        no_message_ref = db.reference('/no_message_data')
        loaded_no_message = no_message_ref.get()
        if loaded_no_message:
            no_message_data.update(loaded_no_message)
            print(f"Loaded {len(no_message_data)} days of no-message data.")
        else:
            print("No existing no-message data found in Firebase.")

        # Load audio counts
        audio_count_ref = db.reference('/audio_counts/count')
        loaded_audio_count = audio_count_ref.get()
        if loaded_audio_count is not None: # Check for None explicitly
             audio_counts["count"] = loaded_audio_count
             print(f"Loaded audio count: {audio_counts['count']}")
        else:
             # Initialize in Firebase if it doesn't exist
             audio_count_ref.set(0)
             audio_counts["count"] = 0
             print("Initialized audio count in Firebase.")

    except Exception as e:
        print(f"Error loading data from Firebase: {e}")
        # Decide how to handle this - continue with empty data, retry, etc.

# --- Firebase Data Saving Functions (Replaced JSON saves) ---
# Note: Firebase keys cannot contain '.', '#', '$', '[', or ']'
# Using MM-DD-YY format for keys instead of MM/DD/YY
def get_firebase_safe_date(dt=None):
    """Returns date as MM-DD-YY string."""
    if dt is None:
        dt = datetime.now(eastern)
    return dt.strftime("%m-%d-%y")

def save_online_time_to_firebase(date_key, detection_key, time_str):
    try:
        ref = db.reference(f'/online_times/{date_key}/{detection_key}')
        ref.set(time_str)
    except Exception as e:
        print(f"Error saving online time to Firebase: {e}")

def increment_no_message_count_in_firebase(date_key):
    try:
        ref = db.reference(f'/no_message_data/{date_key}')
        # Use a transaction for safe incrementing
        def transaction_update(current_value):
            return (current_value or 0) + 1
        ref.transaction(transaction_update)
    except Exception as e:
        print(f"Error incrementing no-message count in Firebase: {e}")

def increment_audio_count_in_firebase():
    try:
        ref = db.reference('/audio_counts/count')
        # Use a transaction for safe incrementing
        def transaction_update(current_value):
            # Ensure current_value is treated as 0 if it's None initially
            return (current_value or 0) + 1
        ref.transaction(transaction_update)
        # Update local cache after successful transaction (optional but good practice)
        # Fetch the new value to be certain, though transaction implies success
        new_count = ref.get()
        if new_count is not None:
            audio_counts["count"] = new_count

    except Exception as e:
        print(f"Error incrementing audio count in Firebase: {e}")


# --- API Call Function (Unchanged) ---
def send_prompt(
    prompt, max_length=50, do_sample=True, top_k=50, top_p=0.95, temperature=1.0
):
    payload = {
        "prompt": prompt, "max_length": max_length, "do_sample": do_sample,
        "top_k": top_k, "top_p": top_p, "temperature": temperature,"key": CHAT_KEY
    }
    try:
        response = requests.post(f"{API_URL}/generate", json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response field found")
        else:
            return f"Error: Received status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "Error: Could not connect to the API." # Return error message
    
def remove_think_blocks(text):
    cleaned = re.sub(r"<think>[\s\S]*?<\/think>", "", text)
    return cleaned.lstrip("\n")
    
def send_think(prompt):

    payload = {
        "key": ALEX_KEY,
        "prompt": prompt,
    }

    try:
        # Send the POST request
        response = requests.post(f"{API_URL}/think", json=payload)

        # Check if the response is successful
        if response.status_code == 200:
            data = response.json()
            cleaned_response = remove_think_blocks(data.get("response", "No response field found"))
            return cleaned_response
        else:
            return f"Error: Received status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# --- Bot Events ---
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    load_data_from_firebase() # Load data when bot starts
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send(
            f"Fired up and ready to lurk! {LURKER} {LURKIN}"
        )
    reset_daily_count.start()


@bot.event
async def on_presence_update(before, after):
    # Ensure USER_ID is correctly compared (it's now an int)
    if after.id == USER_ID:
        today_key = get_firebase_safe_date() # Use Firebase-safe date key
        current_time_str = datetime.now(eastern).strftime("%H:%M")

        # User goes online
        if (
            before.status != discord.Status.online
            and after.status == discord.Status.online
        ):
            user_status[USER_ID]["online"] = True
            user_status[USER_ID]["message_sent"] = False
            regular_transitions[today_key]["online"] += 1 # Still track ephemeral daily

            # Determine next detection key
            # Fetch current detections for today to find the count
            today_detections_ref = db.reference(f'/online_times/{today_key}')
            current_detections = today_detections_ref.get()
            detection_count = len(current_detections) if current_detections else 0
            detection_key = f"detection{detection_count + 1}"

            # Save to Firebase
            save_online_time_to_firebase(today_key, detection_key, current_time_str)

            # Update in-memory cache (important for commands)
            if today_key not in user_online_times:
                 user_online_times[today_key] = {}
            user_online_times[today_key][detection_key] = current_time_str

            # Send Discord message
            channel = discord.utils.get(after.guild.text_channels, name=CHANNEL)
            if channel:
                await channel.send(
                    f"{after.mention} - {detection_count + 1} {LURKIN}"
                )
                await channel.send(
                    "https://cdn.discordapp.com/attachments/1000447385887051827/1293029673470525450/image.png?ex=6705e339&is=670491b9&hm=4c42945052a9e0de8b959dca888a92d25b215aef14ecf58e5805802cb17474d5&"
                )

        # User goes offline
        elif (
            before.status == discord.Status.online
            and after.status != discord.Status.online
        ):
            regular_transitions[today_key]["offline"] += 1
            if (
                user_status[USER_ID]["online"]
                and not user_status[USER_ID]["message_sent"]
            ):
                # Increment no-message count in Firebase and local cache
                increment_no_message_count_in_firebase(today_key)
                no_message_data[today_key] = no_message_data.get(today_key, 0) + 1

                # user_status online_count seems redundant if we have no_message_data
                # If you still need it for some other logic, increment it here:
                # user_status[USER_ID]["online_count"] += 1

            user_status[USER_ID]["online"] = False


@bot.event
async def on_message(message):
    # Ensure USER_ID is correctly compared
    if message.author.id == USER_ID:
        user_status[USER_ID]["message_sent"] = True

    global triggered_messages
    if message.channel:
        try:
            messages = [msg async for msg in message.channel.history(limit=6)]
            messages = [msg for msg in messages if msg.content != TRIPLE_GIF]
            if len(messages) == 6 and all(msg.author == bot.user and msg.id not in triggered_messages for msg in messages):
                await message.channel.send(TRIPLE_GIF)
                triggered_messages.update(msg.id for msg in messages)
        except discord.errors.Forbidden:
            print(f"Missing permissions to read history in channel {message.channel.name}")
        except Exception as e:
            print(f"Error processing message history: {e}")

    await bot.process_commands(message)


# --- Bot Commands ---

@bot.command()
async def audio(ctx):
    if ctx.author.voice:
        voice_channel = ctx.author.voice.channel
        try:
            voice_client = await voice_channel.connect()
        except discord.errors.ClientException:
             await ctx.send("Already connected to a voice channel.")
             return # Avoid error if already connected

        sound_file = AUDIO_FILE
        if not os.path.exists(sound_file):
             await ctx.send("Audio file not found.")
             if voice_client.is_connected():
                 await voice_client.disconnect()
             return

        if not voice_client.is_playing():
            try:
                voice_client.play(
                    discord.FFmpegPCMAudio(sound_file),
                    after=lambda e: print(f"Finished playing: {e}" if e else "Finished playing audio."),
                )

                while voice_client.is_playing():
                    await asyncio.sleep(1)

                # Increment count in Firebase *after* playing finishes
                increment_audio_count_in_firebase()
                # The local audio_counts dict is updated inside the increment function

            except Exception as e:
                print(f"Error playing audio: {e}")
                await ctx.send("An error occurred while trying to play the audio.")
            finally:
                 if voice_client.is_connected():
                    await voice_client.disconnect()

    else:
        # Display the count (fetch from local cache, which should be up-to-date)
        current_count = audio_counts.get("count", 0)
        await ctx.send(f"I hit that shit {current_count} times")


@bot.command(
    name="leaderboard",
    brief="Displays the leaderboard",
    description="Displays the top 5 Alex Detections. Add an optional year to see top 5 for that year.",
    usage="[year (optional)]"
)
async def leaderboard(
    ctx,
    year: int = commands.parameter(
        default=None,
        description="Optional year to filter leaderboard (e.g., 2024)"
    )
):
    # Data is already loaded into user_online_times at startup
    # If you expect data to change frequently *while the bot is running*
    # without restarting, you might want to reload here, but for this use case,
    # relying on the startup load and event updates is likely sufficient.
    # Example reload (optional):
    # online_times_ref = db.reference('/online_times')
    # current_online_times = online_times_ref.get() or {}

    current_online_times = user_online_times # Use in-memory data

    if year:
        year_suffix = str(year)[-2:]
        # Filter keys based on the MM-DD-YY format
        filtered_days = {
            day: detections
            for day, detections in current_online_times.items()
            if day.endswith(f"-{year_suffix}") # Check MM-DD-YY
        }

        if not filtered_days:
            await ctx.send(f"No detection data found for the year {year}.")
            return

        sorted_days = sorted(
            filtered_days.items(),
            key=lambda item: len(item[1]) if isinstance(item[1], dict) else 0, # Handle potential non-dict values
            reverse=True
        )[:5]

        message = f"{LURKER} **Top 5 Detection Days for {year}** {LURKIN}\n"
    else:
        sorted_days = sorted(
            current_online_times.items(),
            key=lambda item: len(item[1]) if isinstance(item[1], dict) else 0, # Handle potential non-dict values
            reverse=True
        )[:5]

        message = f"{LURKER} **Top 5 Detection Days (All Time)** {LURKIN}\n"

    for i, (day, detections) in enumerate(sorted_days, start=1):
        # Ensure detections is a dict before getting len
        count = len(detections) if isinstance(detections, dict) else 0
        message += f"{i}. {day} - {count} Alex detections\n" # Use MM-DD-YY

    await ctx.send(message)


@bot.command(
    name="chat",
    brief="Responds with an Alex complete the sentence",
    description="Responds with a message trained on alex's discord messages",
)
async def chat(
    ctx, *, prompt: str = commands.parameter(default=None, description="A prompt")
):
    if prompt is None:
        await ctx.send("No prompt")
    else:
        # Consider adding async handling if send_prompt takes long
        output = send_prompt(
            prompt=prompt, max_length=50, do_sample=True,
            top_k=40, top_p=0.9, temperature=0.7,
        )
        await ctx.send(output)
        
@bot.command(
    name="think",
    brief="Respond with an attitude like Alex",
    description="Responds with a message that was trained to sound like Alex",
)
async def think(
    ctx, *, prompt: str = commands.parameter(default=None, description="A prompt")
):
    if prompt is None:
        await ctx.send("No prompt")
    else:
        # Consider adding async handling if send_prompt takes long
        output = send_think(prompt=prompt)
        await ctx.send(output)

@bot.command(
    name="alex",
    brief="Displays detections",
    description="Displays number of Alex Detections on a certain date (Format: MM-DD-YY)",
)
async def alex(
    ctx,
    date_str: str = commands.parameter( # Renamed to date_str for clarity
        default=None,
        description="The date to check (Format: MM-DD-YY)",
    ),
):
    if date_str is None:
        date_key = get_firebase_safe_date() # Get today's key
    else:
        # Validate format MM-DD-YY
        try:
            datetime.strptime(date_str, "%m-%d-%y")
            date_key = date_str # Use provided key if valid
        except ValueError:
            await ctx.send("Incorrect date format. Please use MM-DD-YY.")
            return

    # Use in-memory data
    detections_today = user_online_times.get(date_key, {})
    count = len(detections_today) if isinstance(detections_today, dict) else 0

    if count > 0:
        message = f"{LURKIN} {count} detections on {date_key} {LURKER}"
    else:
        message = f"No detections recorded for {date_key}."
    await ctx.send(message)


@bot.command(
    name="nomessage",
    brief="Displays no-message counts",
    description="Shows the no-message counts for a specific date (Format: MM-DD-YY).",
)
async def no_message_count(
    ctx,
    date_str: str = commands.parameter( # Renamed to date_str
        default=None,
        description="The date to check (Format: MM-DD-YY)",
    ),
):
    if date_str is None:
        date_key = get_firebase_safe_date()
    else:
        # Validate format MM-DD-YY
        try:
            datetime.strptime(date_str, "%m-%d-%y")
            date_key = date_str
        except ValueError:
            await ctx.send("Incorrect date format. Please use MM-DD-YY.")
            return

    # Use in-memory data
    count = no_message_data.get(date_key, 0) # Default to 0 if not found
    detections_on_date = user_online_times.get(date_key, {})
    total_detections = len(detections_on_date) if isinstance(detections_on_date, dict) else 0

    if total_detections > 0:
        percentage = round((count / total_detections) * 100, 2) if total_detections > 0 else 0
        await ctx.send(
            f"{LURKER} Detected {count} non verbal lurks on {date_key} ({percentage}% of {total_detections} detections)"
        )
    elif count > 0: # Case where there's a no_message count but no detections (unlikely but possible)
         await ctx.send(f"{LURKER} Detected {count} non verbal lurks on {date_key}, but no corresponding detection entries found.")
    else:
        await ctx.send(f"No no-message lurks recorded for {date_key}.")


@bot.command(
    name="average",
    brief="Displays average detection time delta",
    description="Displays the average time between detections for a specific date (Format: MM-DD-YY).",
    usage="[date (optional, MM-DD-YY)]",
)
async def average_time(
    ctx,
    date_str=commands.parameter( # Renamed to date_str
        default=None,
        description="Specifies which date to get average detections (MM-DD-YY)",
    ),
):
    if date_str is None:
        date_key = get_firebase_safe_date()
    else:
        # Validate format MM-DD-YY
        try:
            datetime.strptime(date_str, "%m-%d-%y")
            date_key = date_str
        except ValueError:
            await ctx.send("Incorrect date format. Please use MM-DD-YY.")
            return

    # Use in-memory data
    detections = user_online_times.get(date_key, {})

    if not isinstance(detections, dict) or len(detections) < 2:
        await ctx.send(f"Not enough detections on {date_key} for an average calculation.")
        return

    # Sort detection times before calculating intervals
    try:
        # Extract times and sort them
        times_str = sorted(detections.values())
        times_dt = [datetime.strptime(t, "%H:%M") for t in times_str]

        if len(times_dt) < 2:
             await ctx.send(f"Not enough valid time entries on {date_key} for an average calculation.")
             return

        intervals = [(times_dt[i + 1] - times_dt[i]).total_seconds() / 60 for i in range(len(times_dt) - 1)]

        # Filter out negative intervals if necessary (e.g., crossing midnight, though unlikely with H:M format)
        intervals = [i for i in intervals if i >= 0]

        if not intervals:
             await ctx.send(f"Could not calculate valid time intervals for {date_key}.")
             return

        avg_interval = sum(intervals) / len(intervals)
        await ctx.send(
            f"Average time between detections on {date_key}: {avg_interval:.2f} minutes"
        )
    except ValueError as e:
         await ctx.send(f"Error parsing time data for {date_key}: {e}")
    except Exception as e:
         await ctx.send(f"An unexpected error occurred during average calculation: {e}")


@bot.command(
    name="graph",
    brief="Displays a graph of detections",
    description="Graphs the amount of detections over the last X days.",
    usage="[days (optional, default 7)]",
)
async def graph_detections(
    ctx,
    days: int = commands.parameter(
        default=7, description="Optional number of days to plot detections for"
    ),
):
    if days <= 0:
        await ctx.send("Number of days must be positive.")
        return

    end_date = datetime.now(eastern)
    # Use in-memory data
    current_online_times = user_online_times

    dates_keys = []
    counts = []
    for i in range(days):
        current_date = end_date - timedelta(days=i)
        date_key = get_firebase_safe_date(current_date) # Get MM-DD-YY key
        dates_keys.append(date_key)
        detections_on_date = current_online_times.get(date_key, {})
        counts.append(len(detections_on_date) if isinstance(detections_on_date, dict) else 0)

    # Reverse lists to plot chronologically
    dates_keys.reverse()
    counts.reverse()

    plt.figure(figsize=(10, 5))
    plt.plot(dates_keys, counts, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Detections Over the Past {days} Days")
    plt.xlabel("Date (MM-DD-YY)")
    plt.ylabel("Detections")
    plt.tight_layout() # Adjust layout

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close() # Close plot to free memory

    await ctx.send(file=discord.File(buf, "detections.png"))
    buf.close()


@bot.command(
    name="graphhourly",
    brief="Displays the average detections per hour in a date range",
    description="Plots a histogram showing the average detections per hour across a specified date range (Format: MM-DD-YY). Defaults to 11-01-24 onwards if no dates are provided.",
    usage="[start_date (optional MM-DD-YY)] [end_date (optional MM-DD-YY)]",
)
async def graphhourly(ctx, start_date_str: str = None, end_date_str: str = None):
    # Use Firebase-safe default date
    default_start_date_key = "11-01-24"
    default_end_date_key = get_firebase_safe_date() # Today

    # Parse input dates (expecting MM-DD-YY)
    try:
        start_date_key = start_date_str if start_date_str else default_start_date_key
        end_date_key = end_date_str if end_date_str else default_end_date_key

        # Validate keys by attempting to parse them
        start_dt = datetime.strptime(start_date_key, "%m-%d-%y")
        end_dt = datetime.strptime(end_date_key, "%m-%d-%y")

    except ValueError:
        await ctx.send("Invalid date format. Please use MM-DD-YY.")
        return

    # Validate date range
    if start_dt > end_dt:
        await ctx.send("Start date must be earlier than or equal to the end date.")
        return

    # Use in-memory data
    current_online_times = user_online_times
    hourly_counts = defaultdict(int)
    total_days_in_range = 0

    # Iterate through dates in the range
    current_dt = start_dt
    while current_dt <= end_dt:
        date_key = get_firebase_safe_date(current_dt)
        if date_key in current_online_times:
            detections = current_online_times[date_key]
            if isinstance(detections, dict): # Ensure it's a dictionary
                total_days_in_range += 1
                for time_str in detections.values():
                    try:
                        hour = datetime.strptime(time_str, "%H:%M").hour
                        hourly_counts[hour] += 1
                    except ValueError:
                        print(f"Skipping invalid time format '{time_str}' on date {date_key}")
        current_dt += timedelta(days=1)


    if total_days_in_range == 0:
        await ctx.send(
            f"No detection data found between {start_date_key} and {end_date_key}."
        )
        return

    # Calculate average detections per hour
    average_detections = [hourly_counts[hour] / total_days_in_range for hour in range(24)]

    # Plotting
    plt.figure(figsize=(12, 6)) # Slightly wider figure
    plt.bar(range(24), average_detections, color="teal", alpha=0.7, edgecolor="black")
    plt.xticks(range(24)) # Ensure all hours are labeled
    plt.title(f"Average Detections Per Hour ({start_date_key} - {end_date_key})")
    plt.xlabel("Hour of the Day (EST)")
    plt.ylabel("Average Detections")
    plt.grid(axis='y', linestyle='--', alpha=0.6) # Add horizontal grid lines
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    await ctx.send(file=discord.File(buf, "average_hourly_detections.png"))
    buf.close()


# --- Tasks ---
@tasks.loop(time=time(0, 0, tzinfo=eastern))
async def reset_daily_count():
    today_key = get_firebase_safe_date() # Use MM-DD-YY
    print(f"Performing daily reset tasks for {today_key}")
    # Reset ephemeral in-memory transition counts
    regular_transitions.clear()
    # user_status online_count reset (if you still use it)
    # user_status[USER_ID]["online_count"] = 0

    # Send reset message
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send(f"{LURKER} - Daily Reset - {LURKIN}")
    else:
        print("Reset message channel not found.")


# --- Run Bot ---
if __name__ == "__main__":
    if not all([BOT_TOKEN, CHANNEL, USER_ID, API_URL, CHANNEL_ID, FIREBASE_KEY_PATH, FIREBASE_DB_URL, AUDIO_FILE]):
        print("Error: Missing one or more required environment variables.")
        exit()
    if not os.path.exists(FIREBASE_KEY_PATH):
        print(f"Error: Firebase service account key not found at {FIREBASE_KEY_PATH}")
        exit()

    bot.run(BOT_TOKEN)
