"""
A tiny program so that i can publish on bsky and X at the
same time (thinking of adding Mastodon also).
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests-oauthlib",
#   "requests",
#   "rich",
#   "atproto",
# ]
# ///

import os
import json
import argparse
from requests_oauthlib import OAuth1Session
from atproto import Client

def xweet(thought):
    x_key = os.environ.get("CONSUMER_KEY")
    x_secret = os.environ.get("CONSUMER_SECRET")
    access_token = os.environ.get("ACCESS_TOKEN")
    access_token_secret = os.environ.get("ACCESS_TOKEN_SECRET")

    payload = {"text": thought}

    oauth = OAuth1Session(
            x_key,
            client_secret=x_secret,
            resource_owner_key=access_token,
            resource_owner_secret=access_token_secret,
            )

    response = oauth.post("https://api.twitter.com/2/tweets", json=payload)

    if response.status_code != 201:
        raise Exception(f"Request returned an error: {response.status_code} {response.text}")

def bsky(thought):
    client = Client()
    client.login(os.environ.get("BLUESKY_HANDLE"), os.environ.get("BLUESKY_PASSWORD"))
    post = client.send_post(thought)

def main():
    parser = argparse.ArgumentParser(description="Crosspost a message to X, Bsky and Mastodon")
    parser.add_argument("thought", help="The text to post")
    args = parser.parse_args()
    xweet(args.thought)
    print("xweet xweeted")
    bsky(args.thought)
    print("bsky flown")


if __name__ == "__main__":
    main()
