# gemini-cli-anthropic

A proxy server that bridges Anthropic's Claude API format to Google's Gemini API, using [gemini-cli](https://github.com/google-gemini/gemini-cli) as a submodule (v0.15.4).

## Setup Instructions

### 1. Clone with submodules

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

### 2. Install dependencies

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

### 3. Run the server

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
- `DEBUG_LOG_REQUESTS`: Set to `true` to enable detailed request/response logging.

## Update Submodule (if needed)

```bash
git submodule update --remote gemini-cli
```

## Submodule Info

- **Path**: `gemini-cli/`
- **Repository**: https://github.com/google-gemini/gemini-cli
- **Version**: v0.15.4 (commit 40fa8136e)
