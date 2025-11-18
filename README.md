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

## Environment Variables

- `PORT`: Server port (default: 3000)
- `GEMINI_API_KEY`: Your Google Gemini API key
- `USE_CCPA`: Set to `true` to use Google Cloud credentials instead of API key
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your Google Cloud credentials file (if using CCPA)
- `DEBUG_LOG_REQUESTS`: Set to `true` to enable debug logging

## Update Submodule (if needed)

```bash
git submodule update --remote gemini-cli
```

## Submodule Info

- **Path**: `gemini-cli/`
- **Repository**: https://github.com/google-gemini/gemini-cli
- **Version**: v0.15.4 (commit 40fa8136e)
