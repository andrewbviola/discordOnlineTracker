import discord
import pytz
from discord.ext import commands, tasks
from datetime import datetime, timedelta, time
from collections import defaultdict
import json
import asyncio
import os
import matplotlib.pyplot as plt
import io
import requests
import re
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DATA_FILE = "online_times.json"
NO_MESSAGE_FILE = "no_message_data.json"
AUDIO_COUNT_FILE = os.getenv("AUDIO_COUNT")
AUDIO_FILE = os.getenv("AUDIO_FILE")

LURKIN = "<:lurkin:1275247450285932565>"
LURKER = "<:lurker:1257490595266560071>"

TRIPLE_GIF="https://tenor.com/view/rodrick-camera-point-rodrick-heffley-diary-of-a-wimpy-kid-gif-13001614153341977897"

CHANNEL = os.getenv("CHANNEL")  # replace with your specific channel
USER_ID = os.getenv("USER_ID")  # replace with specific user ID for tracking
API_URL = os.getenv("API")
CHANNEL_ID=os.getenv("CHANNEL_ID")  # replace with your specific channel ID

CHAT_KEY = os.getenv("CHAT_KEY")
ALEX_KEY = os.getenv("ALEX_KEY")

intents = discord.Intents.all()
intents.presences = True
intents.members = True
triggered_messages = set()

bot = commands.Bot(command_prefix="!", intents=intents)
eastern = pytz.timezone("America/New_York")

# Dictionary to store timestamps of detections for each day
user_online_times = defaultdict(lambda: defaultdict(list))
user_status = defaultdict(
    lambda: {"online": False, "message_sent": False, "online_count": 0}
)
regular_transitions = defaultdict(lambda: {"online": 0, "offline": 0})


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
        response = requests.post(f"{API_URL}/think", json=payload, timeout=300)

        # Check if the response is successful
        if response.status_code == 200:
            data = response.json()
            cleaned_response = remove_think_blocks(data.get("response", "No response field found"))
            return cleaned_response
        else:
            return f"Error: Received status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


# Load no-message data from file if it exists
no_message_data = defaultdict(int)
if os.path.isfile(NO_MESSAGE_FILE):
    with open(NO_MESSAGE_FILE, "r") as file:
        no_message_data.update(json.load(file))
        
audio_counts = defaultdict(int)
audio_counts["count"] = 0
if os.path.isfile(AUDIO_COUNT_FILE):
    with open(AUDIO_COUNT_FILE, "r") as file:
        audio_counts.update(json.load(file))
        
def save_audio_counts():
    with open(AUDIO_COUNT_FILE, "w") as file:
        json.dump(audio_counts, file)


def save_no_message_data():
    with open(NO_MESSAGE_FILE, "w") as file:
        json.dump(no_message_data, file)


# Load detection times from file if it exists
if os.path.isfile(DATA_FILE):
    with open(DATA_FILE, "r") as file:
        user_online_times.update(json.load(file))


def save_data():
    with open(DATA_FILE, "w") as file:
        json.dump(user_online_times, file, default=str)


def get_current_est_date():
    est = pytz.timezone("America/New_York")
    return datetime.now(est).strftime("%m/%d/%y")


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    channel = bot.get_channel(int(CHANNEL_ID))
    if channel:
        await channel.send(
            f"Fired up and ready to lurk! {LURKER} {LURKIN}"
        )
    reset_daily_count.start()  # Start the daily reset task


@bot.event
async def on_presence_update(before, after):
    if after.id == USER_ID:
        today = get_current_est_date()
        current_time = datetime.now(pytz.timezone("America/New_York")).strftime("%H:%M")

        # User goes online
        if (
            before.status != discord.Status.online
            and after.status == discord.Status.online
        ):
            user_status[USER_ID]["online"] = True
            user_status[USER_ID]["message_sent"] = False  # Reset message sent flag
            regular_transitions[today]["online"] += 1  # Track regular online count
            # Record detection
            user_online_times[today][
                f"detection{len(user_online_times[today]) + 1}"
            ] = current_time
            save_data()
            channel = discord.utils.get(after.guild.text_channels, name=CHANNEL)
            if channel:
                await channel.send(
                    f"{after.mention} - {len(user_online_times[today])} {LURKIN}"
                )
                await channel.send(
                    "https://cdn.discordapp.com/attachments/1000447385887051827/1293029673470525450/image.png?ex=6705e339&is=670491b9&hm=4c42945052a9e0de8b959dca888a92d25b215aef14ecf58e5805802cb17474d5&"
                )

        # User goes offline
        elif (
            before.status == discord.Status.online
            and after.status != discord.Status.online
        ):
            regular_transitions[today]["offline"] += 1  # Track regular offline count
            if (
                user_status[USER_ID]["online"]
                and not user_status[USER_ID]["message_sent"]
            ):
                user_status[USER_ID]["online_count"] += 1
                no_message_data[today] += 1  # Increment no-message count for the day
                save_no_message_data()  # Save no-message count
            user_status[USER_ID]["online"] = False  # Reset online status


@bot.event
async def on_message(message):
    if message.author.id == USER_ID:
        user_status[USER_ID]["message_sent"] = True  # Mark as sent a message

    global triggered_messages

    # Check if the last 6 messages are from the bot
    if message.channel:
        # Fetch the last 6 messages
        messages = [msg async for msg in message.channel.history(limit=6)]
        
        # Exclude the most recent GIF message from the count
        messages = [
            msg for msg in messages
            if msg.content != TRIPLE_GIF
        ]
        
        # Check if all remaining 6 messages are from the bot and not already triggered
        if len(messages) == 6 and all(msg.author == bot.user and msg.id not in triggered_messages for msg in messages):
            # Send the GIF
            await message.channel.send(TRIPLE_GIF)
            
            # Add the IDs of these messages to the blacklist
            triggered_messages.update(msg.id for msg in messages)
    await bot.process_commands(message)  # Ensure other commands are processed


@bot.command()
async def audio(ctx):
    if ctx.author.voice:
        voice_channel = ctx.author.voice.channel
        voice_client = await voice_channel.connect()

        # Path to your sound file (update with your own file)
        sound_file = AUDIO_FILE

        # Ensure ffmpeg is installed for audio playback
        if not voice_client.is_playing():
            voice_client.play(
                discord.FFmpegPCMAudio(sound_file),
                after=lambda e: print(f"Finished playing: {e}"),
            )

            # Wait for the sound to finish playing
            while voice_client.is_playing():
                await asyncio.sleep(1)

            await voice_client.disconnect()
            audio_counts["count"] += 1
            save_audio_counts()
    else:
        await ctx.send(f"I hit that shit {audio_counts['count']} times")


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
    # If year is provided, filter days from that year
    if year:
        # Filter days that match the specified year
        # Date format is MM/DD/YY, so we need to check the last 2 digits of the year
        year_suffix = str(year)[-2:]  # Get last 2 digits of year
        filtered_days = {
            day: detections 
            for day, detections in user_online_times.items() 
            if day.endswith(f"/{year_suffix}")
        }
        
        if not filtered_days:
            await ctx.send(f"No detection data found for the year {year}.")
            return
            
        sorted_days = sorted(
            filtered_days.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )[:5]
        
        message = f"{LURKER} **Top 5 Detection Days for {year}** {LURKIN}\n"
    else:
        # All-time leaderboard (original behavior)
        sorted_days = sorted(
            user_online_times.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )[:5]
        
        message = f"{LURKER} **Top 5 Detection Days (All Time)** {LURKIN}\n"
    
    # Generate the leaderboard message
    for i, (day, detections) in enumerate(sorted_days, start=1):
        message += f"{i}. {day} - {len(detections)} Alex detections\n"
    
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
    description="Displays number of Alex Detections on a certain date",
)
async def alex(
    ctx,
    date: str = commands.parameter(
        default=None,
        description="The date to check (Format: MM/DD/YY)",
    ),
):
    if date is None:
        date = get_current_est_date()
    detections_today = user_online_times.get(date, {})
    if detections_today:
        message = f"{LURKIN} {len(detections_today.items())} detections today {LURKER}"
    else:
        message = "No detections recorded today."
    await ctx.send(message)


@bot.command(
    name="nomessage",
    brief="Displays no-message counts",
    description="Shows the no-message counts for a specific date.",
)
async def no_message_count(
    ctx,
    date: str = commands.parameter(
        default=None,
        description="The date to check (Format: MM/DD/YY)",
    ),
):
    if date is None:
        date = get_current_est_date()
    if len(date) != len(get_current_est_date()):
        await ctx.send("Incorrect date format. Please use MM/DD/YY.")
        return
    count = no_message_data.get(date, None)
    detections_today = len(user_online_times.get(date, {}))
    if count is None:
        await ctx.send(f"No data available for {date}.")
    else:
        await ctx.send(
            f"{LURKER} Detected {count} non verbal lurks on {date} ({round((count/detections_today)*100,2)}% of detections)"
        )


@bot.command(
    name="average",
    brief="Displays average detection time delta",
    description="Displays the average time between detections for the day. Add an optional date argument (Format: MM/DD/YY) to find the average of a certain date",
    usage="[date (optional)]",
)
async def average_time(
    ctx,
    date=commands.parameter(
        default=None,
        description="Specifies which date to get average detections (MM/DD/YY)",
    ),
):
    if date is None:
        date = get_current_est_date()
    if len(date) != len(get_current_est_date()):
        await ctx.send("Incorrect date format")
        return
    detections = user_online_times.get(date, {})
    if len(detections) < 2:
        await ctx.send("Not enough detections for an average calculation.")
        return
    times = [datetime.strptime(time, "%H:%M") for time in detections.values()]
    intervals = [(times[i + 1] - times[i]).seconds / 60 for i in range(len(times) - 1)]
    avg_interval = sum(intervals) / len(intervals)
    await ctx.send(
        f"Average time between detections on {date}: {avg_interval:.2f} minutes"
    )


@bot.command(
    name="graph",
    brief="Displays a graph of detections",
    description="Graphs the amount of detections over the last 7 days. Add an optional day argument to plot over an x number of days",
    usage="[days (optional)]",
)
async def graph_detections(
    ctx,
    days: int = commands.parameter(
        default=7, description="Optional number of days to plot detections for"
    ),
):
    end_date = datetime.now(pytz.timezone("America/New_York"))
    start_date = end_date - timedelta(days=int(days))

    dates = [
        (start_date + timedelta(days=i)).strftime("%m/%d/%y")
        for i in range((end_date - start_date).days + 1)
    ]
    counts = [len(user_online_times.get(date, {})) for date in dates]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, counts, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Detections Over the Past {days} Days")
    plt.xlabel("Date")
    plt.ylabel("Detections")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    await ctx.send(file=discord.File(buf, "detections.png"))
    buf.close()


@bot.command(
    name="graphhourly",
    brief="Displays the average detections per hour in a date range",
    description="Plots a histogram showing the average detections per hour across a specified date range. Defaults to 11/1/24 onwards if no dates are provided.",
    usage="[start_date (optional)] [end_date (optional)]",
)
async def graphhourly(ctx, start_date: str = None, end_date: str = None):
    # Default date range
    default_start_date = datetime.strptime("11/01/24", "%m/%d/%y")
    default_end_date = datetime.now()

    # Parse input dates
    try:
        start_date = (
            datetime.strptime(start_date, "%m/%d/%y")
            if start_date
            else default_start_date
        )
        end_date = (
            datetime.strptime(end_date, "%m/%d/%y") if end_date else default_end_date
        )
    except ValueError:
        await ctx.send("Invalid date format. Please use MM/DD/YY.")
        return

    # Validate date range
    if start_date > end_date:
        await ctx.send("Start date must be earlier than or equal to the end date.")
        return

    # Flatten all detections into a single list of hours (within date range)
    hourly_counts = defaultdict(int)
    total_days = 0  # Count valid days

    for date_str, detections in user_online_times.items():
        date_obj = datetime.strptime(date_str, "%m/%d/%y")
        if start_date <= date_obj <= end_date:
            total_days += 1
            for time_str in detections.values():
                hour = datetime.strptime(time_str, "%H:%M").hour
                hourly_counts[hour] += 1

    # Handle edge case if no valid days exist
    if total_days == 0:
        await ctx.send(
            f"No valid data exists for dates between {start_date.strftime('%m/%d/%y')} and {end_date.strftime('%m/%d/%y')}."
        )
        return

    # Calculate the average detections per hour
    average_detections = [hourly_counts[hour] / total_days for hour in range(24)]

    # Plotting the histogram
    plt.figure(figsize=(10, 5))
    plt.bar(range(24), average_detections, color="teal", alpha=0.7, edgecolor="black")
    plt.xticks(range(24))
    plt.title(
        f"Average Detections Per Hour ({start_date.strftime('%m/%d/%y')} - {end_date.strftime('%m/%d/%y')})"
    )
    plt.xlabel("Hour of the Day")
    plt.ylabel("Average Detections")
    plt.tight_layout()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    # Send the plot to the Discord channel
    await ctx.send(file=discord.File(buf, "average_hourly_detections.png"))
    buf.close()


@tasks.loop(time=time(0, 0, tzinfo=eastern))  # This sets the reset time to midnight EST
async def reset_daily_count():
    today = get_current_est_date()
    print(f"Resetting daily online counts for {today}")
    for guild in bot.guilds:
        channel = discord.utils.get(guild.text_channels, name=CHANNEL)
        if channel:
            await channel.send(f"{LURKER} - Reset - {LURKIN}:")
    # Reset daily in-memory counts but keep past data in files
    user_status[USER_ID]["online_count"] = 0


bot.run(BOT_TOKEN)
