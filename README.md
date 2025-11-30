# Qwen3-VL Video Understanding MCP Server (Blaxel)

An MCP (Model Context Protocol) server that enables Claude and other AI agents to analyze videos and images using Qwen3-VL-8B-Instruct deployed on Blaxel.

## Features

- **Video Analysis**: Analyze videos via URL with custom prompts
- **Image Analysis**: Analyze images via URL
- **Video Summarization**: Generate summaries in different styles
- **Text Extraction**: Extract on-screen text and transcribe speech
- **Video Q&A**: Ask specific questions about video content
- **H100 GPUs**: Fast inference on NVIDIA H100 GPUs via Blaxel

## Architecture

```
Claude/Agent → MCP Server → Blaxel API → Qwen3-VL (H100 GPUs)
```

## Prerequisites

1. **Blaxel Account**: Sign up at [blaxel.ai](https://blaxel.ai)
2. **Blaxel CLI**: Install the Blaxel CLI
3. **ffmpeg**: Required for video frame extraction
4. **Python 3.10+**

## Quick Start

### 1. Deploy the Model to Blaxel

```bash
cat << 'EOF' | blaxel apply -f -
apiVersion: blaxel.ai/v1alpha1
kind: Model
metadata:
  name: qwen-qwen3-vl-8b-instruct
  displayName: Qwen/Qwen3-VL-8B-Instruct
spec:
  enabled: true
  policies: []
  flavors:
    - name: nvidia-h100/x4
      type: gpu
  runtime:
    model: Qwen/Qwen3-VL-8B-Instruct
    type: hf_private_endpoint
    image: ''
    args: []
    endpointName: qwenqwen3-vl-8b-instruct-nvidia-h100
    organization: adamanz
  integrationConnections:
    - huggingface-4s2m2h
EOF
```

Or use the provided config:

```bash
blaxel apply -f blaxel-model.yaml
```

### 2. Get Your API Key

```bash
blaxel auth token
```

### 3. Install the MCP Server

```bash
cd qwen-video-blaxel-mcp
pip install -e .
```

Or with uv:

```bash
uv pip install -e .
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Blaxel API key
```

### 5. Add to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "qwen3-video-blaxel": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/qwen-video-blaxel-mcp",
        "run",
        "server.py"
      ],
      "env": {
        "BLAXEL_API_KEY": "your-blaxel-api-key",
        "BLAXEL_MODEL": "qwen-qwen3-vl-8b-instruct"
      }
    }
  }
}
```

### 6. Restart Claude Desktop

The `qwen3-video-blaxel` tools should now be available.

## Available Tools

### `analyze_video`
Analyze a video with a custom prompt.

```python
analyze_video(
  video_url="https://example.com/video.mp4",
  question="What happens in this video?",
  max_frames=8
)
```

### `analyze_image`
Analyze an image with a custom prompt.

```python
analyze_image(
  image_url="https://example.com/image.jpg",
  question="Describe this image"
)
```

### `summarize_video`
Generate a video summary.

```python
summarize_video(
  video_url="https://example.com/video.mp4",
  style="detailed"  # brief, standard, or detailed
)
```

### `video_qa`
Ask specific questions about a video.

```python
video_qa(
  video_url="https://example.com/video.mp4",
  question="How many people appear?"
)
```

### `extract_video_text`
Extract text and transcribe speech.

```python
extract_video_text(
  video_url="https://example.com/presentation.mp4"
)
```

### `check_configuration`
Check the Blaxel API configuration.

### `list_capabilities`
List all server capabilities.

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `BLAXEL_API_KEY` | Your Blaxel API key | Required |
| `BLAXEL_API_URL` | Blaxel API URL | `https://api.blaxel.ai/v1` |
| `BLAXEL_MODEL` | Model name | `qwen-qwen3-vl-8b-instruct` |

## Requirements

- **ffmpeg**: Required for video frame extraction
  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  apt install ffmpeg
  ```

## Supported Formats

**Video**: mp4, webm, mov, avi

**Image**: jpg, jpeg, png, gif, webp

## Comparison: Modal vs Blaxel

| Feature | Modal | Blaxel |
|---------|-------|--------|
| Model | Qwen2.5-VL-7B | Qwen3-VL-8B |
| GPU | A100 | H100 |
| Pricing | Pay-per-second | Subscription |
| Cold Start | ~30-60s | Faster |
| Setup | Deploy code | Apply YAML |

## License

MIT
