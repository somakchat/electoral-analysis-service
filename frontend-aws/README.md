# Political Strategy Maker - AWS Frontend

Production-ready React frontend for the Political Strategy Maker, deployed on AWS CloudFront + S3.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CloudFront CDN                               │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Static Assets (React SPA) ──→ S3 Bucket                         ││
│  │ API Requests (/api/*) ────────→ API Gateway HTTP                ││
│  │ WebSocket (/ws/*) ────────────→ API Gateway WebSocket           ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- 🎨 Modern React UI with Tailwind CSS and Framer Motion
- 🔄 Real-time agent activity visualization via WebSocket
- 📊 Interactive strategy results with SWOT analysis, scenarios, and voter segments
- 📁 Document upload and ingestion
- 💬 Multi-turn conversation with follow-up suggestions
- 📝 Feedback and correction system
- 🌐 Global CDN distribution with CloudFront
- 🔒 Security headers and HTTPS enforcement

## Prerequisites

- **Node.js** 18+ 
- **AWS CLI** configured with credentials
- **AWS SAM CLI** for infrastructure deployment
- Backend API deployed (HTTP + WebSocket endpoints)

## Quick Start

### 1. Install Dependencies

```bash
cd frontend-aws
npm install
```

### 2. Local Development

Create `.env.local`:

```env
VITE_API_URL=http://127.0.0.1:8000
VITE_WS_URL=ws://127.0.0.1:8000/ws/chat
```

Start development server:

```bash
npm run dev
```

### 3. Production Deployment

Deploy to AWS:

```powershell
.\deploy.ps1 `
    -BackendApiUrl "https://your-api.execute-api.ap-south-1.amazonaws.com" `
    -BackendWsUrl "wss://your-ws.execute-api.ap-south-1.amazonaws.com/production" `
    -Environment production `
    -Region ap-south-1
```

## Project Structure

```
frontend-aws/
├── infrastructure/
│   └── template.yaml          # CloudFormation/SAM template
├── src/
│   ├── components/
│   │   ├── Header.tsx         # Navigation header
│   │   ├── ChatInterface.tsx  # Chat UI
│   │   ├── AgentPanel.tsx     # Agent activity display
│   │   ├── StrategyResults.tsx# Strategy visualization
│   │   ├── DocumentUpload.tsx # File upload
│   │   ├── FeedbackPanel.tsx  # User feedback
│   │   └── Sidebar.tsx        # Configuration sidebar
│   ├── hooks/
│   │   ├── useWebSocket.ts    # WebSocket management
│   │   └── useSession.ts      # Session management
│   ├── services/
│   │   └── api.ts             # REST API client
│   ├── types/
│   │   └── index.ts           # TypeScript types
│   ├── styles/
│   │   └── main.css           # Tailwind + custom CSS
│   ├── App.tsx                # Main application
│   └── main.tsx               # Entry point
├── deploy.ps1                 # Deployment script
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## Deployment Options

### Full Deployment (Infrastructure + Code)

```powershell
.\deploy.ps1 -BackendApiUrl "https://..." -BackendWsUrl "wss://..."
```

### Code Only (Skip Infrastructure)

```powershell
.\deploy.ps1 -BackendApiUrl "https://..." -BackendWsUrl "wss://..." -SkipInfra
```

### Build Only (No Deploy)

```bash
npm run build
```

## Infrastructure Resources

The CloudFormation template creates:

| Resource | Description |
|----------|-------------|
| S3 Bucket | Static asset storage |
| CloudFront Distribution | CDN with caching |
| Origin Access Control | Secure S3 access |
| Cache Policies | Static vs API caching |
| Security Headers | CSP, HSTS, etc. |
| CloudWatch Dashboard | Monitoring |
| Logs Bucket | Access logs |

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VITE_API_URL` | Backend HTTP API URL | `https://api.example.com` |
| `VITE_WS_URL` | Backend WebSocket URL | `wss://ws.example.com/production` |
| `VITE_ENVIRONMENT` | Environment name | `production` |

### CloudFormation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `Environment` | Deployment environment | `production` |
| `BackendApiUrl` | Backend API URL | Required |
| `BackendWsUrl` | WebSocket URL | Required |
| `DomainName` | Custom domain (optional) | Empty |

## API Integration

The frontend communicates with the backend via:

### REST Endpoints

- `POST /ingest` - Document upload
- `POST /chat` - Strategy queries
- `POST /chat/interactive` - Interactive chat
- `GET /health` - Health check
- `POST /feedback` - User feedback
- `GET /rag/*` - RAG queries
- `GET /memory/*` - Session memory

### WebSocket

- `ws://*/ws/chat` - Real-time chat with agent streaming

## Customization

### Styling

Edit `src/styles/main.css` and `tailwind.config.js` for:

- Color schemes
- Typography
- Animations
- Component styles

### Adding Components

1. Create component in `src/components/`
2. Import in `App.tsx`
3. Add to layout

### Adding API Endpoints

1. Add types in `src/types/index.ts`
2. Add functions in `src/services/api.ts`
3. Use in components

## Troubleshooting

### WebSocket Connection Issues

1. Check CORS settings in backend
2. Verify WebSocket URL format (`wss://` for production)
3. Check browser console for errors

### Build Errors

1. Clear node_modules and reinstall
2. Check TypeScript errors with `npm run build`
3. Verify all imports are correct

### Deployment Errors

1. Check AWS credentials
2. Verify SAM CLI installation
3. Check CloudFormation events in AWS Console

## Contributing

1. Create feature branch
2. Make changes
3. Test locally with `npm run dev`
4. Build and verify with `npm run build`
5. Submit PR

## License

Proprietary - Political Strategy Maker

