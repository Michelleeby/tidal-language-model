#!/bin/bash

# A script to compile training frames into a video using ffmpeg.

# Get the project root directory
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# --- Configuration ---
DEFAULT_FPS=30
EXPERIMENT_DIR_BASE="${PROJECT_ROOT}/experiments"
FRAMES_SUBDIR="semantic_space_frames"
OUTPUT_FILENAME="semantic_space_evolution.mp4"

# --- Script Logic ---
if [ -z "$1" ]; then
  echo "Usage: $0 <experiment_id> [fps]"
  echo "Example: $0 commit_2edff9d... 30"
  exit 1
fi

EXPERIMENT_ID=$1
FPS=${2:-$DEFAULT_FPS} # Use provided FPS or default to 30

# Construct the full path to the frames
FRAMES_PATH="${EXPERIMENT_DIR_BASE}/${EXPERIMENT_ID}/${FRAMES_SUBDIR}"

if [ ! -d "$FRAMES_PATH" ]; then
  echo "Error: Frames directory not found at ${FRAMES_PATH}"
  exit 1
fi

echo "Source Frames: ${FRAMES_PATH}"
echo "Output Video: ./${OUTPUT_FILENAME}"
echo "Framerate: ${FPS} fps"

# Run ffmpeg to compile the video. The -y flag overwrites the output file if it exists.
ffmpeg -y -framerate "$FPS" -pattern_type glob -i "${FRAMES_PATH}/frame_*.png" -c:v libx264 -pix_fmt yuv420p "$OUTPUT_FILENAME"

echo "✅ Video compilation complete!"