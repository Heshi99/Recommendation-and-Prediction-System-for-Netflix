import os
import time

from flask import Flask

app = Flask(__name__)

# Define the directory to monitor
directory_to_watch = 'C:/Users/SINGER/Desktop/MINI_PROJECT'

# Store the initial modification times of files in the directory
initial_file_modification_times = {}

def get_file_modification_times(directory):
    """
    Get the modification times of files in the specified directory.
    Returns a dictionary mapping file names to modification times.
    """
    modification_times = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            modification_times[filename] = os.path.getmtime(filepath)
    return modification_times

def check_for_file_changes():
    """
    Check for changes in the monitored directory and handle them.
    """
    global initial_file_modification_times
    
    current_file_modification_times = get_file_modification_times(directory_to_watch)
    
    for filename, current_modification_time in current_file_modification_times.items():
        initial_modification_time = initial_file_modification_times.get(filename)
        if initial_modification_time is None:
            # File was added
            print(f"New file added: {filename}")
        elif current_modification_time > initial_modification_time:
            # File was modified
            print(f"File modified: {filename}")
    
    # Update the initial modification times
    initial_file_modification_times = current_file_modification_times

# Route to trigger file change checking
@app.route('/check_file_changes')
def trigger_file_changes():
    check_for_file_changes()
    return 'File changes checked'

if __name__ == '__main__':
    # Initialize initial file modification times
    initial_file_modification_times = get_file_modification_times(directory_to_watch)
    
    # Periodically check for file changes (every 60 seconds)
    while True:
        check_for_file_changes()
        time.sleep(60)  # Adjust the interval as needed
