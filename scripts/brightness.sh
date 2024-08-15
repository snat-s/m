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
    ;;
  decrease)
    # Decrease the brightness
    NEW_BRIGHTNESS=$(echo "$CURRENT_BRIGHTNESS - $STEP" | bc)
    if (( $(echo "$NEW_BRIGHTNESS < 0" | bc -l) )); then
      NEW_BRIGHTNESS=0
    fi
    ;;
  *)
    echo "Invalid action. Use 'increase' or 'decrease'."
    exit 1
    ;;
esac

# Set the new brightness level
xrandr --output $(xrandr | grep " connected" | cut -d" " -f1) --brightness $NEW_BRIGHTNESS

echo "Brightness adjusted to $NEW_BRIGHTNESS"
