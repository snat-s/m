# ASCII Gaussian Splatting

An exploration of Gaussian splatting rendered as ASCII art in the browser.

## What is this?

This is a real-time 3D Gaussian splatting renderer that outputs to ASCII characters instead of traditional graphics. It combines the soft, volumetric look of Gaussian splats with the retro aesthetic of ASCII art.

## Features

- **Multiple shapes**: Gaussian cloud, torus, and wireframe cube
- **Color modes**: Matrix green, amber CRT, cyan glow, white, rainbow
- **ASCII character sets**: Standard gradient, Unicode blocks, binary, dots
- **Interactive controls**: Drag to rotate, scroll to zoom
- **Adjustable parameters**: Rotation speed, splat size, point count

## How it works

1. 3D points are positioned according to Gaussian distributions
2. Points are projected to 2D with perspective
3. Each point is rendered as a soft Gaussian "splat" (radial gradient)
4. The resulting image is sampled and converted to ASCII characters based on brightness
5. Characters are displayed in a retro terminal-style interface

## Usage

Just open `index.html` in a browser.

## The Math

Gaussian splatting represents 3D scenes as collections of 3D Gaussians. Each Gaussian has:
- Position (x, y, z)
- Covariance (size/shape)
- Opacity

When rendered, each Gaussian is projected to 2D and drawn as a soft blob. The overlapping blobs create smooth, continuous surfaces.

This demo simplifies the full Gaussian splatting pipeline but captures the essential visual effect: soft, volumetric 3D rendering that looks great even when converted to ASCII.
