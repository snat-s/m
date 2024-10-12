#!/bin/bash

# Get the current brightness level
CURRENT_BRIGHTNESS=$(xrandr --verbose | grep -i brightness | cut -d ' ' -f2)

# Define the step to adjust the brightness
STEP=0.1

# Check the argument to determine the action
ACTION=$1

case $ACTION in
  increase)
    # Increase the brightness
    NEW_BRIGHTNESS=$(echo "$CURRENT_BRIGHTNESS + $STEP" | bc)
    if (( $(echo "$NEW_BRIGHTNESS > 1" | bc -l) )); then
      NEW_BRIGHTNESS=1
    fi
    action_msg="Brightness Up"
    ;;
  decrease)
    # Decrease the brightness
    NEW_BRIGHTNESS=$(echo "$CURRENT_BRIGHTNESS - $STEP" | bc)
    if (( $(echo "$NEW_BRIGHTNESS < 0" | bc -l) )); then
      NEW_BRIGHTNESS=0
    fi
    action_msg="Brightness Down"
    ;;
  *)
    echo "Invalid action. Use 'increase' or 'decrease'."
    exit 1
    ;;
esac

# Set the new brightness level
xrandr --output $(xrandr | grep " connected" | cut -d" " -f1) --brightness $NEW_BRIGHTNESS

# Convert brightness to percentage for notification
BRIGHTNESS_PERCENT=$(echo "$NEW_BRIGHTNESS * 100" | bc | cut -d. -f1)

# Display a notification with the current brightness
notify-send "$action_msg" "Current brightness: $BRIGHTNESS_PERCENT%" -h int:value:$BRIGHTNESS_PERCENT -h string:synchronous:brightness -i none

echo "Brightness adjusted to $NEW_BRIGHTNESS"
