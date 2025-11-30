"""
Qwen3-VL Video Understanding MCP Server (Blaxel)

An MCP server that uses Qwen3-VL-8B-Instruct deployed on Blaxel to analyze
videos and images. Provides video understanding capabilities to Claude Code
and other AI agents.

Features:
- Analyze videos via URL
- Analyze images via URL
- Multiple analysis modes (summary, detailed, Q&A)
- Uses Blaxel's OpenAI-compatible API
"""

import os
import base64
import asyncio
import httpx
from typing import Optional, Annotated
from datetime import timedelta

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Initialize the MCP server
mcp = FastMCP("qwen3-video-understanding")

# Configuration
BLAXEL_API_URL = os.environ.get(
    "BLAXEL_API_URL",
    "https://run.blaxel.ai/simple/models/qwen-qwen3-vl-8b-instruct-3/v1"
)
BLAXEL_API_KEY = os.environ.get("BLAXEL_API_KEY", "")
BLAXEL_MODEL = os.environ.get(
    "BLAXEL_MODEL",
    "Qwen/Qwen3-VL-8B-Instruct"
)

# Timeout settings
REQUEST_TIMEOUT = 300  # 5 minutes


def get_headers() -> dict:
    """Get API headers with authentication."""
    headers = {"Content-Type": "application/json"}
    if BLAXEL_API_KEY:
        headers["Authorization"] = f"Bearer {BLAXEL_API_KEY}"
    return headers


async def download_and_encode_image(url: str) -> str:
    """Download image and return base64 encoded data."""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")


async def extract_video_frames_from_url(video_url: str, max_frames: int = 8) -> list:
    """
    Extract frames from video URL using ffmpeg.
    Returns list of base64 encoded frame images.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    frames = []

    try:
        # Create temp directory for frames
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download video first
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.get(video_url)
                response.raise_for_status()

                video_path = Path(tmpdir) / "video.mp4"
                video_path.write_bytes(response.content)

            # Extract frames using ffmpeg
            frame_pattern = str(Path(tmpdir) / "frame_%04d.jpg")

            # Get video duration first
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)

            try:
                duration = float(result.stdout.strip())
            except:
                duration = 60  # Default to 60 seconds if can't detect

            # Calculate frame interval
            fps = max_frames / duration if duration > 0 else 1

            extract_cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"fps={fps}",
                "-frames:v", str(max_frames),
                "-q:v", "2",
                frame_pattern
            ]
            subprocess.run(extract_cmd, capture_output=True, timeout=120)

            # Read and encode frames
            for i in range(1, max_frames + 1):
                frame_path = Path(tmpdir) / f"frame_{i:04d}.jpg"
                if frame_path.exists():
                    frame_data = frame_path.read_bytes()
                    frames.append(base64.b64encode(frame_data).decode("utf-8"))

    except Exception as e:
        print(f"Error extracting frames: {e}")

    return frames


async def call_blaxel_vision(
    images: list,
    prompt: str,
    max_tokens: int = 1024
) -> dict:
    """
    Call Blaxel's Qwen3-VL model with images.

    Uses OpenAI-compatible chat completions API with vision.
    """
    # Build message content with images
    content = []

    for img_data in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_data}"
            }
        })

    content.append({
        "type": "text",
        "text": prompt
    })

    payload = {
        "model": BLAXEL_MODEL,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{BLAXEL_API_URL}/chat/completions",
                json=payload,
                headers=get_headers()
            )
            response.raise_for_status()
            result = response.json()

            return {
                "status": "success",
                "answer": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {})
            }

    except httpx.TimeoutException:
        return {"status": "error", "error": "Request timed out"}
    except httpx.HTTPStatusError as e:
        return {"status": "error", "error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
async def analyze_video(
    video_url: Annotated[str, Field(description="URL of the video to analyze (must be publicly accessible)")],
    question: Annotated[str, Field(description="Question or prompt about the video")] = "Describe what happens in this video in detail.",
    max_frames: Annotated[int, Field(description="Maximum number of frames to extract (1-16)")] = 8,
    max_tokens: Annotated[int, Field(description="Maximum tokens in response")] = 1024
) -> dict:
    """
    Analyze a video using Qwen3-VL-8B vision-language model on Blaxel.

    The video must be accessible via a public URL. The model will:
    1. Download the video
    2. Extract key frames (up to max_frames)
    3. Analyze the frames with your question

    Examples:
    - "What happens in this video?"
    - "Summarize the main events"
    - "What products are shown?"
    - "Describe the people and their actions"
    """
    try:
        # Extract frames from video
        frames = await extract_video_frames_from_url(
            video_url,
            max_frames=min(max(1, max_frames), 16)
        )

        if not frames:
            return {
                "status": "error",
                "error": "Could not extract frames from video. Ensure ffmpeg is installed.",
                "video_url": video_url
            }

        # Build prompt for video analysis
        full_prompt = f"""These are {len(frames)} frames extracted from a video in chronological order.

{question}

Analyze the visual content across all frames to understand what happens in the video."""

        result = await call_blaxel_vision(frames, full_prompt, max_tokens)

        if result["status"] == "success":
            return {
                "status": "success",
                "video_url": video_url,
                "question": question,
                "frames_analyzed": len(frames),
                "analysis": result["answer"],
                "usage": result.get("usage", {})
            }
        else:
            return {
                "status": "error",
                "error": result.get("error", "Unknown error"),
                "video_url": video_url
            }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "video_url": video_url
        }


@mcp.tool()
async def analyze_image(
    image_url: Annotated[str, Field(description="URL of the image to analyze (must be publicly accessible)")],
    question: Annotated[str, Field(description="Question or prompt about the image")] = "Describe this image in detail.",
    max_tokens: Annotated[int, Field(description="Maximum tokens in response")] = 512
) -> dict:
    """
    Analyze an image using Qwen3-VL-8B vision-language model on Blaxel.

    The image must be accessible via a public URL.

    Examples:
    - "What's in this image?"
    - "Describe the scene"
    - "What text is visible?"
    - "Identify any people or objects"
    """
    try:
        # Download and encode image
        img_data = await download_and_encode_image(image_url)

        result = await call_blaxel_vision([img_data], question, max_tokens)

        if result["status"] == "success":
            return {
                "status": "success",
                "image_url": image_url,
                "question": question,
                "analysis": result["answer"],
                "usage": result.get("usage", {})
            }
        else:
            return {
                "status": "error",
                "error": result.get("error", "Unknown error"),
                "image_url": image_url
            }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "image_url": image_url
        }


@mcp.tool()
async def summarize_video(
    video_url: Annotated[str, Field(description="URL of the video to summarize")],
    style: Annotated[str, Field(description="Summary style: 'brief', 'standard', or 'detailed'")] = "standard"
) -> dict:
    """
    Generate a summary of a video.

    Styles:
    - brief: 1-2 sentence overview
    - standard: 1-2 paragraph summary
    - detailed: Comprehensive analysis
    """
    prompts = {
        "brief": "Provide a 1-2 sentence summary of what happens in this video.",
        "standard": "Summarize this video in 1-2 paragraphs. Include the main topic, key events, and overall message.",
        "detailed": """Provide a comprehensive analysis:
1. Main topic/theme
2. Key events in order
3. Important visual elements
4. Overall takeaway"""
    }

    prompt = prompts.get(style, prompts["standard"])
    max_tokens = {"brief": 128, "standard": 512, "detailed": 1024}.get(style, 512)
    max_frames = {"brief": 6, "standard": 8, "detailed": 12}.get(style, 8)

    return await analyze_video(
        video_url=video_url,
        question=prompt,
        max_frames=max_frames,
        max_tokens=max_tokens
    )


@mcp.tool()
async def video_qa(
    video_url: Annotated[str, Field(description="URL of the video")],
    question: Annotated[str, Field(description="Your specific question about the video")]
) -> dict:
    """
    Ask a specific question about a video's content.

    Examples:
    - "How many people appear?"
    - "What color is the car?"
    - "What is being demonstrated?"
    """
    return await analyze_video(
        video_url=video_url,
        question=f"Answer this question: {question}\n\nProvide a clear, direct answer.",
        max_frames=8,
        max_tokens=512
    )


@mcp.tool()
async def extract_video_text(
    video_url: Annotated[str, Field(description="URL of the video")]
) -> dict:
    """Extract text and transcribe speech from a video."""
    return await analyze_video(
        video_url=video_url,
        question="""Extract all text content:
1. On-screen text, titles, captions
2. Transcribe spoken words
3. Text from documents or slides shown
List each with context.""",
        max_frames=12,
        max_tokens=1024
    )


@mcp.tool()
def check_configuration() -> dict:
    """Check the Blaxel API configuration."""
    return {
        "status": "configured",
        "api_url": BLAXEL_API_URL,
        "model": BLAXEL_MODEL,
        "api_key_set": bool(BLAXEL_API_KEY),
        "notes": [
            "Requires BLAXEL_API_KEY environment variable",
            "Video analysis requires ffmpeg installed locally",
            "Model: Qwen3-VL-8B-Instruct on H100 GPUs"
        ]
    }


@mcp.tool()
def list_capabilities() -> dict:
    """List all server capabilities."""
    return {
        "model": "Qwen3-VL-8B-Instruct",
        "deployment": "Blaxel (H100 GPUs)",
        "capabilities": [
            "Video summarization",
            "Video Q&A",
            "Image analysis",
            "Text extraction",
            "Multi-frame analysis"
        ],
        "supported_formats": {
            "video": ["mp4", "webm", "mov", "avi"],
            "image": ["jpg", "jpeg", "png", "gif", "webp"]
        },
        "requirements": {
            "BLAXEL_API_KEY": "Required for authentication",
            "ffmpeg": "Required for video frame extraction"
        }
    }


@mcp.resource("resource://server-info")
def get_server_info() -> dict:
    """Get server information."""
    return {
        "name": "Qwen3 Video Understanding MCP (Blaxel)",
        "version": "1.0.0",
        "description": "Analyze videos/images using Qwen3-VL on Blaxel",
        "model": "Qwen3-VL-8B-Instruct",
        "backend": "Blaxel H100 GPUs"
    }


def main():
    """Main entry point."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
