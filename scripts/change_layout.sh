#!/bin/bash

# Check the current layout
CURRENT_LAYOUT=$(setxkbmap -query | grep layout | awk '{print $2}')

# Switch to the other layout with options to prevent key swapping
if [ "$CURRENT_LAYOUT" == "us" ]; then
    setxkbmap es -option caps:swapescape
    echo "Switched to ES layout"
else
    setxkbmap us -option caps:swapescape -option altwin:swap_alt_win
    echo "Switched to US layout"
fi
