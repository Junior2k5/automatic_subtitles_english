import re
from datetime import timedelta


def adjust_timestamp(timestamp, delta):
    """Adjust the timestamp by the given delta in seconds."""
    # Parse the timestamp
    hours, minutes, seconds, milliseconds = map(int, re.split(r'[:,]', timestamp))
    time = timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)
    # Adjust the time
    adjusted_time = time + delta
    if adjusted_time.total_seconds() < 0:
        adjusted_time = timedelta(0)
    # Format the adjusted timestamp back to string
    total_seconds = int(adjusted_time.total_seconds())
    milliseconds = int(adjusted_time.microseconds / 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def adjust_srt_timestamps(input_file, output_file, delta_seconds):
    """Adjust the timestamps in the given .srt file."""
    delta = timedelta(seconds=delta_seconds)
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file, 'w', encoding='utf-8') as file:
        for line in lines:
            match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", line)
            if match:
                start_time = match.group(1)
                end_time = match.group(2)
                adjusted_start_time = adjust_timestamp(start_time, delta)
                adjusted_end_time = adjust_timestamp(end_time, delta)
                file.write(f"{adjusted_start_time} --> {adjusted_end_time}\n")
            else:
                file.write(line)


# Usage
input_file = 'input.srt'
output_file = 'output.srt'
adjust_srt_timestamps(input_file, output_file, -31)
