# Swedish Sovereign AI - Demo Web App

Side-by-side comparison of vanilla Mistral-7B vs Riksbanken fine-tuned model.

## Prerequisites

- Node.js 18+
- Modal account with deployed inference server
- Firebase account (for production deployment)

## Local Development

1. Install dependencies:
   ```bash
   cd web
   npm install
   ```

2. Copy `.env.example` to `.env` and set your Modal API URL:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your Modal endpoint URL:
   ```
   VITE_MODAL_API_URL=https://YOUR-WORKSPACE--swedish-sovereign-ai-serve-inferenceserver-chat.modal.run
   ```

3. Deploy the Modal inference server (from project root):
   ```bash
   # Temporary (stops when you Ctrl+C)
   pipenv run modal serve src/serve/modal_serve.py

   # Persistent (stays running)
   pipenv run modal deploy src/serve/modal_serve.py
   ```

   The deploy command will output your endpoint URLs.

4. Start the dev server:
   ```bash
   npm run dev
   ```

   Open http://localhost:5173

## Deployment to Firebase Hosting

### First-time Setup

1. Install Firebase CLI:
   ```bash
   npm install -g firebase-tools
   ```

2. Login to Firebase:
   ```bash
   firebase login
   ```

3. Create a new Firebase project:
   - Go to https://console.firebase.google.com
   - Click "Create a project" (or "Add project")
   - Name it (e.g., `swedish-sovereign-ai`)
   - Disable Google Analytics (optional, faster setup)
   - Once created, click **Hosting** → **Get started** to enable it

4. Initialize Firebase in the web directory:
   ```bash
   firebase init hosting
   ```

   When prompted:
   - Select "Use an existing project" → choose your project
   - Public directory: `dist`
   - Single-page app: `Yes`
   - Set up GitHub deploys: `No`
   - Overwrite dist/index.html: `No`

### Deploy

1. Build with production Modal URL:
   ```bash
   VITE_MODAL_API_URL=https://YOUR-WORKSPACE--swedish-sovereign-ai-serve-inferenceserver-chat.modal.run npm run build
   ```

2. Deploy:
   ```bash
   firebase deploy --only hosting
   ```

Your app will be live at `https://YOUR-SITE-ID.web.app`

### Switching Projects

```bash
# List available projects
firebase projects:list

# Switch to a different project
firebase use PROJECT_ID
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VITE_MODAL_API_URL` | Modal chat endpoint URL (ends with `-chat.modal.run`) |

## Architecture

```
┌──────────────────────┐         ┌─────────────────────────────┐
│  Static Web App      │  ──────►│  Modal Inference Server     │
│  (Firebase Hosting)  │  POST   │  (GPU serverless)           │
│                      │         │                             │
│  - React + TypeScript│         │  - Mistral-7B base model    │
│  - Tailwind CSS      │         │  - LoRA fine-tuned model    │
│  - Vite build        │         │  - /chat endpoint           │
└──────────────────────┘         └─────────────────────────────┘
```

## Features

- **Warmup screen**: Automatically warms up GPU servers on page load
- **Side-by-side comparison**: See vanilla vs fine-tuned responses
- **Clickable examples**: Pre-filled questions that showcase the fine-tuned model
- **Swedish UI**: All copy in Swedish with proper ÅÄÖ
