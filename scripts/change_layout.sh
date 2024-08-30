#!/bin/bash

# Check the current layout
CURRENT_LAYOUT=$(setxkbmap -query | grep layout | awk '{print $2}')

# Switch to the other layout with options to prevent key swapping
if [ "$CURRENT_LAYOUT" == "us" ]; then
    setxkbmap latam -option caps:swapescape
    NEW_LAYOUT="ES"
else
    setxkbmap us -option caps:swapescape 
    NEW_LAYOUT="US"
fi

notify-send "Keyboard Layout" "Switched to $NEW_LAYOUT layout" -i input-keyboard -t 2000

