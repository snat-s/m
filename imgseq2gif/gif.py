import os
import argparse
from PIL import Image

def main(input_dir, output_path):
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")], key=lambda x: int(x.split(".")[0]))

    first_image = Image.open(os.path.join(input_dir, image_files[0]))
    width, height = first_image.size

    frames = []

    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        frame = Image.open(image_path)
        frames.append(frame)

    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=1000/24, loop=0)

    print(f"GIF created: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF from PNG images")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing the PNG images")
    parser.add_argument("-o", "--output", default="output.gif", help="Output file path (default: output.gif)")

    args = parser.parse_args()

    main(args.input_dir, args.output)
