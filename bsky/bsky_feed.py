# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
# ]
# ///
import requests

def main(name="snats.xyz", output_file="following.txt"):
    r = requests.get(f'https://public.api.bsky.app/xrpc/app.bsky.graph.getFollows?actor={name}')
    body = r.json()
    handles = [user['handle'] for user in body['follows']]
    rss_handles = [f"https://bsky.app/profile/{handle}/rss" for handle in handles]
    with open(output_file, 'w') as f:
        f.write('\n'.join(rss_handles))

if __name__ == "__main__":
    main()

