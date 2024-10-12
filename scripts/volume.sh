#!/bin/bash

# Function to get current volume
get_volume() {
    pactl get-sink-volume @DEFAULT_SINK@ | awk '{print $5}' | sed 's/%//'
}

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [up|down]"
    exit 1
fi

# Set the step size for volume change
STEP=5

# Get the current volume before adjustment
current_volume=$(get_volume)

# Adjust volume based on argument
case $1 in
    up)
        # Check if increasing volume would exceed 100%
        if [ $((current_volume + STEP)) -le 100 ]; then
            pactl set-sink-volume @DEFAULT_SINK@ +${STEP}%
            action="Volume Up"
        else
            # Set volume to 100% if it would exceed
            pactl set-sink-volume @DEFAULT_SINK@ 100%
            action="Volume Max"
        fi
        ;;
    down)
        pactl set-sink-volume @DEFAULT_SINK@ -${STEP}%
        action="Volume Down"
        ;;
    *)
        echo "Invalid argument. Use 'up' or 'down'."
        exit 1
        ;;
esac

# Get the current volume after adjustment
current_volume=$(get_volume)

# Display a notification with the current volume
notify-send "$action" "Current volume: $current_volume%" -h int:value:$current_volume -h string:synchronous:volume -i none

# Optional: Uncomment the next line if you want to print the volume in the terminal as well
echo "Current volume: $current_volume%"
