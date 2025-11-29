# gemini-cli-anthropic

Proxy server that enables Claude Code to use Gemini Pro models via [gemini-cli](https://github.com/google-gemini/gemini-cli), leveraging Google's affordable monthly subscription plans. Bridges Anthropic's Claude API format to Google's Gemini API.

## Setup Instructions

### Quick Install (Recommended)

Install globally via npm:

```bash
npm install -g @vitorcen/gemini-cli-anthropic
```

Start the server:

```bash
gemini-cli-anthropic
```

The server will start on port 3000 (configurable via `PORT` environment variable).

### Install from Source

#### 1. Clone with submodules

#### Option 1: Clone with submodules in one command

```bash
git clone --recurse-submodules git@github.com:vitorcen/gemini-cli-anthropic.git
cd gemini-cli-anthropic
```

#### Option 2: Clone then initialize submodules

```bash
git clone git@github.com:vitorcen/gemini-cli-anthropic.git
cd gemini-cli-anthropic
git submodule update --init --recursive
```

#### 2. Install dependencies

Install main project dependencies:

```bash
npm install
```

Install gemini-cli submodule dependencies:

```bash
cd gemini-cli
npm install
cd ..
```

#### 3. Run the server

```bash
npm start
```

The server will listen on port 3000 by default (configurable via `PORT` environment variable).

## Authentication

By default, the server authenticates using Google Cloud Platform (GCP) credentials (`USE_CCPA=true`). Ensure your environment is configured for GCP authentication (e.g., by running `gcloud auth application-default login`).

Alternatively, you can use a Gemini API key by setting the `GEMINI_API_KEY` environment variable.

### Environment Variables

- `PORT`: Server port (default: 3000)
- `GEMINI_API_KEY`: Your Google Gemini API key. If provided, this will be used for authentication instead of the default GCP method.
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your Google Cloud credentials file (if needed).
- `DEBUG_LOG`: Set to `true` to enable detailed request/response logging.

## Usage with Claude Code

Once the proxy server is running, configure Claude Code to use Gemini models by setting these environment variables:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:41242 \
ANTHROPIC_MODEL=gemini-3-pro-preview \
ANTHROPIC_DEFAULT_OPUS_MODEL=gemini-3-pro-preview \
ANTHROPIC_DEFAULT_SONNET_MODEL=gemini-2.5-flash \
ANTHROPIC_DEFAULT_HAIKU_MODEL=gemini-2.5-flash \
claude
```

This configuration:

- Routes all Claude API calls through the proxy server
- Maps Opus requests to Gemini 2.5 Pro (high capability)
- Maps Sonnet/Haiku requests to Gemini 2.5 Flash (fast, cost-effective)

You can customize the model mappings based on your needs and available Gemini models.

## Update Submodule (if needed)

```bash
git submodule update --remote gemini-cli
```

## Submodule Info

- **Path**: `gemini-cli/`
- **Repository**: https://github.com/google-gemini/gemini-cli
- **Version**: v0.15.4 (commit 40fa8136e)
