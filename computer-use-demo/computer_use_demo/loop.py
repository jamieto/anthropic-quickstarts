"""
Agentic sampling loop that calls the Claude API and local implementation of anthropic-defined computer use tools.
"""

import json
import os
import platform
import time
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional, cast

import logging

logger = logging.getLogger(__name__)

import aiomysql
import httpx
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from .tools import (
    TOOL_GROUPS_BY_VERSION,
    ToolCollection,
    ToolResult,
    ToolVersion,
)

from .tools.subagent import SubAgentTool
from .tools.credential_tool import CredentialsTool

PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"

S3_BUCKET = os.getenv('S3_BUCKET', 'your-default-bucket')
CHAT_ID = os.getenv('CHAT_ID', 'default-chat-id')
PROJECT_STATUS_PATH = "/home/computeruse/project/project_status.md"
PROJECT_LOG_PATH = "/home/computeruse/project/project_log.md"

def _load_project_memory() -> str:
    """Load current project context - status first, then recent log entries."""
    memory = ""
    
    # Always load status (small, critical)
    if os.path.exists(PROJECT_STATUS_PATH):
        try:
            with open(PROJECT_STATUS_PATH, "r") as f:
                content = f.read()
                memory += f"""
<PROJECT_STATUS>
{content}
</PROJECT_STATUS>
"""
        except Exception as e:
            logger.error(f"Failed to read project status: {e}")
    
    # Load log (may be larger, but important for context)
    if os.path.exists(PROJECT_LOG_PATH):
        try:
            with open(PROJECT_LOG_PATH, "r") as f:
                content = f.read()
                # Optional: truncate to last N entries if file gets huge
                memory += f"""
<PROJECT_LOG>
{content}
</PROJECT_LOG>
"""
        except Exception as e:
            logger.error(f"Failed to read project log: {e}")
    
    return memory

def _ensure_project_dir():
    """Ensure the project directory exists."""
    project_dir = os.path.dirname(PROJECT_STATUS_PATH)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)

# def _get_deployment_info() -> str:
#     """Get deployment info from environment variables."""
#     app_url = os.getenv('APP_URL')
#     app_port = os.getenv('APP_PORT', '3000')
#     public_url = os.getenv('PUBLIC_URL')
    
#     if not app_url:
#         return ""
    
#     return f"""
# <USER_ACCESSIBLE_ENDPOINT>
# **MAKING THINGS ACCESSIBLE TO THE USER:**

# You can make any HTTP-based service accessible to the user via their browser!

# **Your Public Endpoint:** {app_url}
# **Port to bind:** {app_port}
# **Base Path:** /webapp/

# **How it works:**
# 1. Start any HTTP server on port {app_port} (bind to 0.0.0.0)
# 2. It becomes instantly accessible at {app_url}
# 3. User can access it directly in their browser

# **CRITICAL: BASE PATH HANDLING**

# Your app is served at `/webapp/` - NOT at the root `/`. This means:
# - User visits: `{app_url}/page` 
# - Your server receives: `/page` (prefix stripped)
# - But ALL links/assets in your HTML must account for the `/webapp/` base path!

# **THE PROBLEM:**
# ```html
# <!-- WRONG - These will break! -->
# <a href="/about">About</a>           → goes to domain.com/about (404!)
# <img src="/images/logo.png">         → goes to domain.com/images/logo.png (404!)

# <!-- CORRECT - Use relative paths -->
# <a href="./about">About</a>          → goes to domain.com/webapp/about
# <img src="./images/logo.png">        → goes to domain.com/webapp/images/logo.png
# ```

# **SOLUTION 1: Use HTML Base Tag (Recommended for static sites)**
# Add this to your HTML `<head>`:
# ```html
# <head>
#     <base href="/webapp/">
#     <!-- Now all relative URLs resolve from /webapp/ -->
# </head>
# ```

# **SOLUTION 2: Use Relative Paths**
# Always use `./` prefix for links and assets:
# ```html
# <a href="./about">About</a>
# <link rel="stylesheet" href="./css/style.css">
# <script src="./js/app.js"></script>
# <img src="./images/logo.png">
# ```

# **SOLUTION 3: Framework-Specific Configuration**

# **React (Create React App / Vite):**
# ```javascript
# // vite.config.js
# export default defineConfig({{
#   base: '/webapp/',
# }})

# // Or for Create React App, set in package.json:
# // "homepage": "/webapp/"
# ```

# **Vue:**
# ```javascript
# // vite.config.js
# export default defineConfig({{
#   base: '/webapp/',
# }})
# ```

# **Next.js:**
# ```javascript
# // next.config.js
# module.exports = {{
#   basePath: '/webapp',
#   assetPrefix: '/webapp/',
# }}
# ```

# **Svelte/SvelteKit:**
# ```javascript
# // svelte.config.js
# export default {{
#   kit: {{
#     paths: {{
#       base: '/webapp'
#     }}
#   }}
# }}
# ```

# **Flask:**
# ```python
# from flask import Flask
# app = Flask(__name__)
# app.config['APPLICATION_ROOT'] = '/webapp'

# # Or use url_for() which handles this automatically
# ```

# **Express.js:**
# ```javascript
# const express = require('express');
# const app = express();

# // Mount everything under /webapp awareness
# app.use((req, res, next) => {{
#   res.locals.basePath = '/webapp';
#   next();
# }});
# ```

# **SOLUTION 4: JavaScript Router Configuration**

# **React Router:**
# ```javascript
# <BrowserRouter basename="/webapp">
#   <Routes>
#     <Route path="/" element={{<Home />}} />
#     <Route path="/about" element={{<About />}} />
#   </Routes>
# </BrowserRouter>
# ```

# **Vue Router:**
# ```javascript
# const router = createRouter({{
#   history: createWebHistory('/webapp/'),
#   routes: [...]
# }})
# ```

# **IMPORTANT RULES:**
# 1. ALWAYS use relative paths (`./`) or configure base path
# 2. NEVER use absolute paths starting with `/` for internal links
# 3. For static HTML sites, ALWAYS add `<base href="/webapp/">`
# 4. For frameworks, ALWAYS configure the base path in the config
# 5. Test navigation after deployment - click through all links!

# **CHECKLIST BEFORE DEPLOYMENT:**
# - [ ] HTML has `<base href="/webapp/">` OR framework base path configured
# - [ ] All `<a href>` links use relative paths or respect base
# - [ ] All `<img src>` use relative paths
# - [ ] All `<link href>` use relative paths  
# - [ ] All `<script src>` use relative paths
# - [ ] CSS `url()` references use relative paths
# - [ ] JavaScript fetch/API calls use relative paths
# - [ ] Router configured with base path (if using SPA)

# **QUICK TEST:**
# After starting your server, mentally trace each link:
# - Does `./about` resolve to `/webapp/about`? ✅
# - Does `/about` resolve to `/about`? ❌ (This is wrong!)

# **COMMON USE CASES:**

# **Static Website / HTML Files:**
# ```bash
# # Add base tag to all HTML files first!
# cd /home/computeruse/project/website
# python3 -m http.server {app_port} --bind 0.0.0.0 &
# echo "View at: {app_url}"
# ```

# **React/Vue/Vite Dev Server:**
# ```bash
# cd /home/computeruse/project/frontend
# # Make sure vite.config.js has base: '/webapp/'
# npm run dev -- --host 0.0.0.0 --port {app_port} &
# echo "Dev server at: {app_url}"
# ```

# **Express/Node.js:**
# ```bash
# cd /home/computeruse/project/app
# PORT={app_port} node server.js &
# echo "Server running at: {app_url}"
# ```

# **DEBUGGING:**
# If links are broken after deployment:
# 1. Open browser dev tools (F12)
# 2. Check Network tab for 404 errors
# 3. Look at the failed URL - does it have `/webapp/`?
# 4. Fix the source link to use relative path or add base tag
# </USER_ACCESSIBLE_ENDPOINT>
# """

# Add near the top with other env vars
PLATFORM_BROKER_URL = os.getenv('PLATFORM_BROKER_URL', 'http://platform-broker.default.svc.cluster.local')
BUILD_BROKER_URL = os.getenv('BUILD_BROKER_URL', 'http://build-broker.default.svc.cluster.local:8002')
AGENT_ID = os.getenv('AGENT_ID', 'unknown')
CHAT_ID = os.getenv('CHAT_ID', 'unknown')

def _get_deployment_capabilities() -> str:
    """Get deployment/infrastructure provisioning instructions."""
    return f"""
<INFRASTRUCTURE_PROVISIONING>
**CREATING AND MANAGING DEPLOYMENTS (VCLUSTERS):**

You can provision isolated Kubernetes environments (vclusters) for deploying applications.
Deployment info is stored in shared directories so ALL agents can access it.

**Platform Broker URL:** {PLATFORM_BROKER_URL}
**Build Broker URL:** {BUILD_BROKER_URL}

## SHARED DEPLOYMENT STORAGE

All deployment configurations are stored in the shared project directory:
```
/home/computeruse/project/.deployments/
├── registry.json                    # List of all deployments
├── v-abc123/
│   ├── kubeconfig.yaml             # Kubernetes config  
│   ├── config.json                 # Domain, DB, ECR info
│   └── status.json                 # Current status
└── v-def456/
    └── ...
```

## BEFORE CREATING A NEW DEPLOYMENT

**ALWAYS check for existing deployments first:**
```bash
# Check if deployments directory and registry exist
DEPLOYMENTS_DIR="/home/computeruse/project/.deployments"

if [ -f "$DEPLOYMENTS_DIR/registry.json" ]; then
  echo "=== Existing Deployments ==="
  cat "$DEPLOYMENTS_DIR/registry.json" | jq .
  
  # Get active deployment
  ACTIVE_ID=$(cat "$DEPLOYMENTS_DIR/registry.json" | jq -r '.active // empty')
  
  if [ -n "$ACTIVE_ID" ]; then
    echo ""
    echo "Active deployment: $ACTIVE_ID"
    echo "Loading it now..."
    source "$DEPLOYMENTS_DIR/use-deployment.sh" "$ACTIVE_ID"
  fi
else
  echo "No existing deployments found."
fi
```

## CREATING A NEW DEPLOYMENT (FULL SCRIPT)

**Copy and run this entire script to create a new deployment:**
```bash
#!/bin/bash
set -e

PLATFORM_BROKER_URL="{PLATFORM_BROKER_URL}"
DEPLOYMENTS_DIR="/home/computeruse/project/.deployments"
AGENT_ID="{AGENT_ID}"
CHAT_ID="{CHAT_ID}"

echo "=== Creating New Deployment ==="

# Step 1: Create deployments directory
mkdir -p "$DEPLOYMENTS_DIR"

# Step 2: Request new deployment
echo "Requesting deployment from Platform Broker..."
CREATE_RESPONSE=$(curl -s -X POST "$PLATFORM_BROKER_URL/deployments" \\
  -H "Content-Type: application/json" \\
  -d '{{"description": "Deployment for Chat '$CHAT_ID'"}}')

echo "Response: $CREATE_RESPONSE"

DEPLOYMENT_ID=$(echo "$CREATE_RESPONSE" | jq -r '.id')
STATUS=$(echo "$CREATE_RESPONSE" | jq -r '.status')

if [ -z "$DEPLOYMENT_ID" ] || [ "$DEPLOYMENT_ID" = "null" ]; then
  echo "ERROR: Failed to create deployment"
  echo "$CREATE_RESPONSE"
  exit 1
fi

echo "Deployment ID: $DEPLOYMENT_ID"
echo "Initial status: $STATUS"

# Step 3: Create deployment directory
mkdir -p "$DEPLOYMENTS_DIR/$DEPLOYMENT_ID"

# Step 4: Poll until ready (timeout after 5 minutes)
echo "Waiting for deployment to be ready..."
TIMEOUT=300
ELAPSED=0
INTERVAL=10

while [ $ELAPSED -lt $TIMEOUT ]; do
  STATUS_RESPONSE=$(curl -s "$PLATFORM_BROKER_URL/deployments/$DEPLOYMENT_ID")
  STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
  
  echo "  Status: $STATUS (elapsed: ${{ELAPSED}}s)"
  
  if [ "$STATUS" = "ready" ]; then
    echo "Deployment is ready!"
    ACCESS_TOKEN=$(echo "$STATUS_RESPONSE" | jq -r '.access_token')
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "ERROR: Deployment failed!"
    echo "$STATUS_RESPONSE" | jq .
    exit 1
  fi
  
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done

if [ "$STATUS" != "ready" ]; then
  echo "ERROR: Deployment timed out after ${{TIMEOUT}}s"
  exit 1
fi

# Step 5: Get kubeconfig and credentials
echo "Retrieving kubeconfig and credentials..."
CREDS_RESPONSE=$(curl -s -X POST "$PLATFORM_BROKER_URL/deployments/$DEPLOYMENT_ID/kubeconfig" \\
  -H "Authorization: Bearer $ACCESS_TOKEN")

# Check if we got valid response
if echo "$CREDS_RESPONSE" | jq -e '.kubeconfig' > /dev/null 2>&1; then
  echo "Credentials retrieved successfully!"
else
  echo "ERROR: Failed to get credentials"
  echo "$CREDS_RESPONSE"
  exit 1
fi

# Step 6: Save kubeconfig (IMPORTANT: jq -r for proper YAML formatting)
echo "$CREDS_RESPONSE" | jq -r '.kubeconfig' > "$DEPLOYMENTS_DIR/$DEPLOYMENT_ID/kubeconfig.yaml"
echo "Saved: $DEPLOYMENTS_DIR/$DEPLOYMENT_ID/kubeconfig.yaml"

# Step 7: Save config.json
echo "$CREDS_RESPONSE" | jq '{{
  deployment_id: .deployment_id,
  domain: .domain,
  wildcard_domain: .wildcard_domain,
  base_domain: .base_domain,
  database: .database,
  ecr: .ecr,
  efs: .efs,
  infrastructure: .infrastructure
}}' > "$DEPLOYMENTS_DIR/$DEPLOYMENT_ID/config.json"
echo "Saved: $DEPLOYMENTS_DIR/$DEPLOYMENT_ID/config.json"

# Step 8: Save status.json
cat > "$DEPLOYMENTS_DIR/$DEPLOYMENT_ID/status.json" << EOF
{{
  "status": "ready",
  "created_at": "$(date -Iseconds)",
  "created_by": "$AGENT_ID"
}}
EOF
echo "Saved: $DEPLOYMENTS_DIR/$DEPLOYMENT_ID/status.json"

# Step 9: Update registry.json
DOMAIN=$(echo "$CREDS_RESPONSE" | jq -r '.domain')

if [ -f "$DEPLOYMENTS_DIR/registry.json" ]; then
  # Update existing registry
  TMP_FILE=$(mktemp)
  cat "$DEPLOYMENTS_DIR/registry.json" | jq \\
    --arg id "$DEPLOYMENT_ID" \\
    --arg domain "$DOMAIN" \\
    --arg created_by "$AGENT_ID" \\
    '.deployments += [{{id: $id, domain: $domain, created_at: (now | todate), created_by: $created_by}}] | .active = $id' \\
    > "$TMP_FILE" && mv "$TMP_FILE" "$DEPLOYMENTS_DIR/registry.json"
else
  # Create new registry
  cat > "$DEPLOYMENTS_DIR/registry.json" << EOF
{{
  "active": "$DEPLOYMENT_ID",
  "deployments": [
    {{
      "id": "$DEPLOYMENT_ID",
      "domain": "$DOMAIN",
      "created_at": "$(date -Iseconds)",
      "created_by": "$AGENT_ID"
    }}
  ]
}}
EOF
fi
echo "Updated: $DEPLOYMENTS_DIR/registry.json"

# Step 10: Create helper script (if not exists)
if [ ! -f "$DEPLOYMENTS_DIR/use-deployment.sh" ]; then
cat > "$DEPLOYMENTS_DIR/use-deployment.sh" << 'HELPER_SCRIPT'
#!/bin/bash
# Usage: source /home/computeruse/project/.deployments/use-deployment.sh [deployment_id]

DEPLOYMENTS_DIR="/home/computeruse/project/.deployments"

# Get deployment ID from argument or active deployment
if [ -n "$1" ]; then
  DEPLOYMENT_ID="$1"
else
  if [ -f "$DEPLOYMENTS_DIR/registry.json" ]; then
    DEPLOYMENT_ID=$(cat "$DEPLOYMENTS_DIR/registry.json" | jq -r '.active // empty')
  fi
fi

if [ -z "$DEPLOYMENT_ID" ]; then
  echo "ERROR: No deployment specified and no active deployment found."
  echo "Usage: source use-deployment.sh <deployment_id>"
  return 1 2>/dev/null || exit 1
fi

CONFIG_FILE="$DEPLOYMENTS_DIR/$DEPLOYMENT_ID/config.json"
KUBECONFIG_FILE="$DEPLOYMENTS_DIR/$DEPLOYMENT_ID/kubeconfig.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: Deployment $DEPLOYMENT_ID not found at $CONFIG_FILE"
  return 1 2>/dev/null || exit 1
fi

if [ ! -f "$KUBECONFIG_FILE" ]; then
  echo "ERROR: Kubeconfig not found at $KUBECONFIG_FILE"
  return 1 2>/dev/null || exit 1
fi

# Export KUBECONFIG
export KUBECONFIG="$KUBECONFIG_FILE"

# Export deployment variables
export DEPLOYMENT_ID="$DEPLOYMENT_ID"
export DEPLOY_DOMAIN=$(cat "$CONFIG_FILE" | jq -r '.domain')
export DEPLOY_WILDCARD_DOMAIN=$(cat "$CONFIG_FILE" | jq -r '.wildcard_domain')
export DEPLOY_BASE_DOMAIN=$(cat "$CONFIG_FILE" | jq -r '.base_domain')
export DEPLOY_DB_HOST=$(cat "$CONFIG_FILE" | jq -r '.database.host')
export DEPLOY_DB_PORT=$(cat "$CONFIG_FILE" | jq -r '.database.port')
export DEPLOY_DB_NAME=$(cat "$CONFIG_FILE" | jq -r '.database.name')
export DEPLOY_DB_USER=$(cat "$CONFIG_FILE" | jq -r '.database.username')
export DEPLOY_DB_PASS=$(cat "$CONFIG_FILE" | jq -r '.database.password')
export DEPLOY_ECR_REGISTRY=$(cat "$CONFIG_FILE" | jq -r '.ecr.registry')
export DEPLOY_ECR_PREFIX=$(cat "$CONFIG_FILE" | jq -r '.ecr.prefix')
export DEPLOY_ECR_REGION=$(cat "$CONFIG_FILE" | jq -r '.ecr.region')

echo "════════════════════════════════════════════════════════════"
echo "  DEPLOYMENT LOADED: $DEPLOYMENT_ID"
echo "════════════════════════════════════════════════════════════"
echo "  Domain:      https://$DEPLOY_DOMAIN"
echo "  Database:    $DEPLOY_DB_NAME @ $DEPLOY_DB_HOST"
echo "  ECR:         $DEPLOY_ECR_REGISTRY"
echo "  KUBECONFIG:  $KUBECONFIG"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Verify connection:"
echo "  kubectl get pods"
echo "  kubectl get secrets"
echo ""
HELPER_SCRIPT
chmod +x "$DEPLOYMENTS_DIR/use-deployment.sh"
echo "Created: $DEPLOYMENTS_DIR/use-deployment.sh"
fi

# Step 11: Load the deployment
echo ""
echo "=== Loading Deployment ==="
source "$DEPLOYMENTS_DIR/use-deployment.sh" "$DEPLOYMENT_ID"

# Step 12: Verify connection
echo ""
echo "=== Verifying Kubernetes Connection ==="
if kubectl get pods 2>/dev/null; then
  echo "✓ Connection successful!"
else
  echo "✗ Connection failed - kubeconfig may need refresh"
fi

echo ""
echo "=== Deployment Complete ==="
echo "Domain: https://$DEPLOY_DOMAIN"
echo ""
echo "To use this deployment later, run:"
echo "  source /home/computeruse/project/.deployments/use-deployment.sh"
```

## USING AN EXISTING DEPLOYMENT

**Quick load (uses active deployment):**
```bash
source /home/computeruse/project/.deployments/use-deployment.sh
```

**Load specific deployment:**
```bash
source /home/computeruse/project/.deployments/use-deployment.sh v-6eac0f
```

**After loading, these variables are available:**
- `$DEPLOYMENT_ID` - The deployment ID (e.g., v-6eac0f)
- `$DEPLOY_DOMAIN` - Main domain (e.g., v-6eac0f.evolvesystem1.com)
- `$DEPLOY_DB_HOST` - Database hostname
- `$DEPLOY_DB_NAME` - Database name
- `$DEPLOY_DB_USER` - Database username
- `$DEPLOY_DB_PASS` - Database password
- `$DEPLOY_ECR_REGISTRY` - ECR registry URL
- `$KUBECONFIG` - Path to kubeconfig file

---

## BUILDING DOCKER IMAGES

**Build Broker URL:** {BUILD_BROKER_URL}

When you need to deploy an application, you must first build a Docker image using the Build Broker.
The Build Broker uses Kaniko to build images inside the cluster (no Docker daemon needed).

### BUILD DIRECTORY

Put your source code and Dockerfile in: `/home/computeruse/build/`
```
/home/computeruse/build/
├── Dockerfile
├── src/
│   └── ... your source code ...
├── package.json (if Node.js)
├── requirements.txt (if Python)
└── ... other files ...
```

### COMPLETE BUILD WORKFLOW

**Step 1: Prepare your application code**
```bash
# Create build directory
mkdir -p /home/computeruse/build

# Example: Create a simple Node.js app
cat > /home/computeruse/build/package.json << 'EOF'
{{
  "name": "myapp",
  "version": "1.0.0",
  "main": "server.js",
  "scripts": {{
    "start": "node server.js"
  }},
  "dependencies": {{
    "express": "^4.18.2"
  }}
}}
EOF

cat > /home/computeruse/build/server.js << 'EOF'
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {{
  res.json({{ message: 'Hello from MyApp!', timestamp: new Date() }});
}});

app.get('/health', (req, res) => {{
  res.json({{ status: 'healthy' }});
}});

app.listen(PORT, '0.0.0.0', () => {{
  console.log(`Server running on port ${{PORT}}`);
}});
EOF
```

**Step 2: Create Dockerfile**
```bash
cat > /home/computeruse/build/Dockerfile << 'EOF'
FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install --production

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# Start the app
CMD ["npm", "start"]
EOF
```

**Step 3: Build the image**
```bash
#!/bin/bash
set -e

BUILD_BROKER_URL="{BUILD_BROKER_URL}"
APP_NAME="myapp"

echo "=== Building Docker Image ==="

# Trigger build
# source_path must be "build" to match /home/computeruse/build/ however if you have multiple apps you can structure differently with subdirectories but it must start from "build" e.g. "source_path": "build/myapp"
echo "Submitting build request..."
BUILD_RESPONSE=$(curl -s -X POST "$BUILD_BROKER_URL/builds" \\
  -H "Content-Type: application/json" \\
  -d '{{"name": "'$APP_NAME'", "source_path": "build"}}')

echo "Response: $BUILD_RESPONSE"

BUILD_ID=$(echo "$BUILD_RESPONSE" | jq -r '.build_id')

if [ -z "$BUILD_ID" ] || [ "$BUILD_ID" = "null" ]; then
  echo "ERROR: Failed to start build"
  echo "$BUILD_RESPONSE"
  exit 1
fi

echo "Build ID: $BUILD_ID"

# Poll until complete
echo "Building image (this may take a few minutes)..."
while true; do
  BUILD_STATUS=$(curl -s "$BUILD_BROKER_URL/builds/$BUILD_ID")
  STATUS=$(echo "$BUILD_STATUS" | jq -r '.status')
  
  echo "  Status: $STATUS"
  
  if [ "$STATUS" = "completed" ]; then
    IMAGE_TAG=$(echo "$BUILD_STATUS" | jq -r '.image_tag')
    echo ""
    echo "✓ Build completed!"
    echo "  Image: $IMAGE_TAG"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo ""
    echo "✗ Build failed!"
    echo "Logs:"
    echo "$BUILD_STATUS" | jq -r '.logs'
    exit 1
  fi
  
  sleep 5
done

# Export for use in deployment
export APP_IMAGE="$IMAGE_TAG"
echo ""
echo "Image ready: $APP_IMAGE"
```

**Step 4: Deploy to Kubernetes**
```bash
# Make sure deployment is loaded
source /home/computeruse/project/.deployments/use-deployment.sh

# Deploy the built image
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $APP_NAME
spec:
  replicas: 2
  selector:
    matchLabels:
      app: $APP_NAME
  template:
    metadata:
      labels:
        app: $APP_NAME
    spec:
      serviceAccountName: agent-s3-sa
      containers:
      - name: $APP_NAME
        image: $APP_IMAGE
        ports:
        - containerPort: 3000
        env:
        - name: PORT
          value: "3000"
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DB_HOST
        - name: DB_DATABASE
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DB_DATABASE
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DB_USERNAME
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DB_PASSWORD
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 15
          periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: $APP_NAME
spec:
  selector:
    app: $APP_NAME
  ports:
  - port: 80
    targetPort: 3000
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: $APP_NAME-ingress
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - $DEPLOY_DOMAIN
    secretName: wildcard-tls
  rules:
  - host: $DEPLOY_DOMAIN
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: $APP_NAME
            port:
              number: 80
EOF

echo ""
echo "=== Deployment Complete ==="
echo "Application URL: https://$DEPLOY_DOMAIN"
echo ""
echo "Check status:"
echo "  kubectl get pods -l app=$APP_NAME"
echo "  kubectl logs -l app=$APP_NAME"
```

### FULL END-TO-END SCRIPT

**Complete script to build and deploy an application:**
```bash
#!/bin/bash
set -e

# Configuration
APP_NAME="myapp"
BUILD_BROKER_URL="{BUILD_BROKER_URL}"
DEPLOYMENTS_DIR="/home/computeruse/project/.deployments"

echo "════════════════════════════════════════════════════════════"
echo "  BUILD AND DEPLOY: $APP_NAME"
echo "════════════════════════════════════════════════════════════"

# Step 1: Check for existing deployment or create new one
echo ""
echo "=== Step 1: Checking Deployment ==="
if [ -f "$DEPLOYMENTS_DIR/registry.json" ]; then
  ACTIVE_ID=$(cat "$DEPLOYMENTS_DIR/registry.json" | jq -r '.active // empty')
  if [ -n "$ACTIVE_ID" ]; then
    echo "Using existing deployment: $ACTIVE_ID"
    source "$DEPLOYMENTS_DIR/use-deployment.sh" "$ACTIVE_ID"
  fi
fi

if [ -z "$DEPLOYMENT_ID" ]; then
  echo "No deployment found. Please create one first:"
  echo "  See: CREATING A NEW DEPLOYMENT section"
  exit 1
fi

# Step 2: Verify build directory exists
echo ""
echo "=== Step 2: Checking Build Directory ==="
if [ ! -f "/home/computeruse/build/Dockerfile" ]; then
  echo "ERROR: No Dockerfile found in /home/computeruse/build/"
  echo "Please create your application code and Dockerfile first."
  exit 1
fi
echo "✓ Build directory ready"
ls -la /home/computeruse/build/

# Step 3: Build the image
echo ""
echo "=== Step 3: Building Docker Image ==="
BUILD_RESPONSE=$(curl -s -X POST "$BUILD_BROKER_URL/builds" \\
  -H "Content-Type: application/json" \\
  -d '{{"name": "'$APP_NAME'", "source_path": "build"}}')

BUILD_ID=$(echo "$BUILD_RESPONSE" | jq -r '.build_id')

if [ -z "$BUILD_ID" ] || [ "$BUILD_ID" = "null" ]; then
  echo "ERROR: Failed to start build"
  echo "$BUILD_RESPONSE"
  exit 1
fi

echo "Build ID: $BUILD_ID"
echo "Building... (this may take a few minutes)"

while true; do
  BUILD_STATUS=$(curl -s "$BUILD_BROKER_URL/builds/$BUILD_ID")
  STATUS=$(echo "$BUILD_STATUS" | jq -r '.status')
  
  if [ "$STATUS" = "completed" ]; then
    APP_IMAGE=$(echo "$BUILD_STATUS" | jq -r '.image_tag')
    echo "✓ Build completed: $APP_IMAGE"
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "✗ Build failed!"
    echo "$BUILD_STATUS" | jq -r '.logs'
    exit 1
  fi
  
  echo "  Status: $STATUS"
  sleep 5
done

# Step 4: Deploy to Kubernetes
echo ""
echo "=== Step 4: Deploying to Kubernetes ==="

kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $APP_NAME
spec:
  replicas: 2
  selector:
    matchLabels:
      app: $APP_NAME
  template:
    metadata:
      labels:
        app: $APP_NAME
    spec:
      serviceAccountName: agent-s3-sa
      containers:
      - name: $APP_NAME
        image: $APP_IMAGE
        ports:
        - containerPort: 3000
        env:
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DB_HOST
        - name: DB_DATABASE
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DB_DATABASE
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DB_USERNAME
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DB_PASSWORD
---
apiVersion: v1
kind: Service
metadata:
  name: $APP_NAME
spec:
  selector:
    app: $APP_NAME
  ports:
  - port: 80
    targetPort: 3000
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: $APP_NAME-ingress
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - $DEPLOY_DOMAIN
    secretName: wildcard-tls
  rules:
  - host: $DEPLOY_DOMAIN
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: $APP_NAME
            port:
              number: 80
EOF

# Step 5: Wait for rollout
echo ""
echo "=== Step 5: Waiting for Deployment ==="
kubectl rollout status deployment/$APP_NAME --timeout=120s

# Step 6: Show status
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DEPLOYMENT SUCCESSFUL!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Application URL: https://$DEPLOY_DOMAIN"
echo "  Image: $APP_IMAGE"
echo ""
echo "  Commands:"
echo "    kubectl get pods -l app=$APP_NAME"
echo "    kubectl logs -l app=$APP_NAME -f"
echo ""
```

### EXAMPLE DOCKERFILES

**Python Flask:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

**Python FastAPI:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Node.js Express:**
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install --production
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
```

**Go:**
```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o main .

FROM alpine:3.18
WORKDIR /app
COPY --from=builder /app/main .
EXPOSE 8080
CMD ["./main"]
```

**Static Website (Nginx):**
```dockerfile
FROM nginx:alpine
COPY . /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## REFRESHING EXPIRED KUBECONFIG

Kubeconfig tokens expire after ~1 hour. To refresh:
```bash
DEPLOYMENT_ID="v-6eac0f"  # Your deployment ID
DEPLOYMENTS_DIR="/home/computeruse/project/.deployments"
PLATFORM_BROKER_URL="{PLATFORM_BROKER_URL}"

# Get fresh access token
STATUS_RESPONSE=$(curl -s "$PLATFORM_BROKER_URL/deployments/$DEPLOYMENT_ID")
ACCESS_TOKEN=$(echo "$STATUS_RESPONSE" | jq -r '.access_token')

if [ -z "$ACCESS_TOKEN" ] || [ "$ACCESS_TOKEN" = "null" ]; then
  echo "ERROR: Could not get access token. Deployment may not exist."
  exit 1
fi

# Get fresh kubeconfig
CREDS_RESPONSE=$(curl -s -X POST "$PLATFORM_BROKER_URL/deployments/$DEPLOYMENT_ID/kubeconfig" \\
  -H "Authorization: Bearer $ACCESS_TOKEN")

# Save it
echo "$CREDS_RESPONSE" | jq -r '.kubeconfig' > "$DEPLOYMENTS_DIR/$DEPLOYMENT_ID/kubeconfig.yaml"

echo "Kubeconfig refreshed!"

# Reload deployment
source "$DEPLOYMENTS_DIR/use-deployment.sh" "$DEPLOYMENT_ID"
```

## PRE-CREATED RESOURCES IN EACH VCLUSTER

When you connect to a vcluster, these resources already exist:

| Resource Type | Name | Description |
|--------------|------|-------------|
| Secret | `database-credentials` | DB_HOST, DB_PORT, DB_DATABASE, DB_USERNAME, DB_PASSWORD |
| PVC | `efs-pvc` | Shared EFS storage |
| ServiceAccount | `agent-s3-sa` | Has S3 access via IRSA |
| Secret | `wildcard-tls` | TLS certificate for ingress |

## IMPORTANT RULES

1. **ALWAYS check for existing deployments** before creating new ones
2. **ALWAYS use `jq -r`** when extracting kubeconfig (handles newlines properly)
3. **Put build files in `/home/computeruse/build/`** - this is where the Build Broker looks
4. **Access tokens expire in 5 minutes** - get kubeconfig promptly
5. **Kubeconfig tokens expire in 1 hour** - refresh if kubectl fails
6. **Use `source use-deployment.sh`** to load deployments (sets all variables)
7. **All agents share** the `.deployments` directory - check what exists first
</INFRASTRUCTURE_PROVISIONING>
"""

# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = """<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open Firefox, click on the Firefox icon. Note: firefox-esr is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps may take time to appear. Take a screenshot to confirm.
* When using bash commands that output very large quantities of text, redirect into a tmp file and use str_replace_based_edit_tool or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page. Make sure you scroll down to see everything before deciding something isn't available.
* When using computer function calls, they take a while to run and send back to you. Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is """ + datetime.today().strftime('%A, %B %-d, %Y') + """.
</SYSTEM_CAPABILITY>

<DIRECTORY_STRUCTURE>
Your workspace has six directories with clear purposes:

**/home/computeruse/uploads/** - USER'S INPUT FILES
   - Files the user uploaded for this project
   - When user says "I uploaded a file" or "use the file I gave you" or there are files in /uploads/, check here
   - All team members can access these (read-only)
   - Examples: requirements, data files, images, specifications, reference documents
   - Your starting materials

**/home/computeruse/work/** - YOUR PERSONAL WORKSPACE
   - Your private work-in-progress area
   - Other team members CANNOT see this
   - Persists across your work sessions
   - Use for: research notes, drafts, experiments, personal planning
   - When ready, move polished work to /project/

**/home/computeruse/project/** - TEAM'S SHARED WORKSPACE
   - All team members can see and contribute
   - This is where the actual work happens - both the deliverable AND supporting work
   - Organize in subdirectories that make sense for your task
   
   **Typical organization:**
   - The actual deliverable (website, report, design, whatever you're building)
   - Supporting work (research, planning, documentation, notes)
   - Shared resources (data, images, templates specific to this project)
   
   **Examples:**
   - Web project: /project/webapp/, /project/research/, /project/planning/
   - Report: /project/report.pdf, /project/data_analysis/, /project/sources/
   - Design: /project/designs/, /project/drafts/, /project/client_feedback/
   
   **This is collaborative:**
   - Build upon what other team members created
   - Keep it organized so others can understand your work
   - The deliverable being built here is what the user will receive

**/home/computeruse/library/** - USER'S PERSONAL LIBRARY
   - User's files available across ALL their projects (not just this one)
   - READ-ONLY: you can use but not modify
   - Examples: company logos, brand guidelines, reusable templates
   - Check here if user mentions "my logo", "our brand", "my template" or similar
   - Less common than /uploads/ (which is project-specific)

**/home/computeruse/tools/** - SHARED TOOLS
   - Pre-installed tools available to everyone
   - Explore before creating new tools: ls /home/computeruse/tools/
   - You CAN create new tools in /tools/your_tool_name/
   - You CANNOT modify existing tools (create a new one instead)
   - Each tool has its own subdirectory with README.md

**/home/computeruse/tmp/** - TEMPORARY SCRATCH SPACE
   - Ephemeral - cleared when session ends
   - Use for: downloads, temporary extractions, quick tests, packaging
   - Don't save important work here or work that can be used later as evidence or reference

**/home/computeruse/build/** - BUILD CONTEXT
   - Use this directory when building Docker images
   - Put your Dockerfile and all source code here
   - The Build Broker will look here to build images

**WORKFLOW:**
1. User uploads files → /home/computeruse/uploads/
2. You work privately → /home/computeruse/work/ (drafts, experiments)
3. You share with team → /home/computeruse/project/ (the actual deliverable + supporting work)
4. When complete → package from /home/computeruse/project/ and upload to S3

**VISIBILITY:**
- /home/computeruse/uploads/ - All team members can READ
- /home/computeruse/work/ - Only YOU can access
- /home/computeruse/project/ - All team members can READ and WRITE
- /home/computeruse/library/ - All team members can READ
- /home/computeruse/tools/ - All team members can READ and CREATE new tools
- /home/computeruse/tmp/ - Only YOU can access
</DIRECTORY_STRUCTURE>

<CREDENTIALS_ACCESS>
**SHARED CREDENTIALS:**

The admin has configured shared credentials that you can use for various services.
Use the `credentials` tool to access them:

**List available credentials:**
```
credentials(action="list")
```

**Get a specific credential:**
```
credentials(action="get", slug="openai_api")
```

**SECURITY RULES:**
1. NEVER write credentials to files - use them directly in memory
2. NEVER include credentials in your responses to the user
3. NEVER echo or print credentials to stdout in a way user can see
4. Use credentials directly in API calls, scripts, or login flows
5. If a credential doesn't exist, ask the user to have admin add it

**Common credential types:**
- `api_key`: API keys for services (OpenAI, Stripe, etc.)
- `login`: Username/password for websites
- `oauth`: OAuth client credentials
- `smtp`: Email sending credentials
- `database`: Database connection credentials
- `ssh`: SSH access credentials

**Example workflow:**
1. List credentials to see what's available
2. Get the specific credential you need
3. Use it directly in your API call or script
4. Never save the credential to disk
</CREDENTIALS_ACCESS>

<TOOL_SPECIFICATIONS>
You have 3 tools available:

**1. bash** - Execute shell commands
   - Use for: running commands, installing packages, system operations
   - Example: {"command": "ls -la /home/computeruse/project/"}
   - Timeout: 120 seconds. If timeout occurs, restart with {"restart": true}

**2. str_replace_editor** - Create, view, and edit files
   - **create**: Create new file (REQUIRES file_text with complete content)
   - **view**: View file or directory contents
   - **str_replace**: Replace text (old_str must exist exactly once)
   - **insert**: Insert text at line number (requires insert_line and new_str)
   - **undo_edit**: Undo last edit
   - All paths must be absolute (start with /)

**3. computer** - GUI interaction
   - screenshot, mouse_move, left_click, right_click, type, key, scroll

**TOOL SELECTION:**
- Creating/editing files → str_replace_editor (NOT bash heredoc)
- Running commands → bash
- GUI interaction → computer
</TOOL_SPECIFICATIONS>

<TOOL_SELECTION_STRATEGY>
WHEN TO USE EACH TOOL:

BASH TOOL - Use for:
- Running commands (ls, grep, find, etc.)
- Installing packages (apt-get, pip)
- Running scripts that already exist
- Quick one-liners
- System operations
DO NOT USE FOR: Creating large files (use str_replace_editor instead)

STR_REPLACE_EDITOR TOOL - Use for:
- Creating new files with content
- Editing existing files
- Viewing file contents
- ANY file creation task
BEST FOR: Files of any size, especially large files

COMPUTER TOOL - Use for:
- GUI interactions
- Taking screenshots
- Mouse/keyboard control
- Browser automation

CRITICAL: For creating files with content, ALWAYS use str_replace_editor tool, 
NOT bash with heredoc. The str_replace_editor tool is specifically designed 
for this and has no size limits.
</TOOL_SELECTION_STRATEGY>

<TOOL_MANAGEMENT>
**FINDING EXISTING TOOLS:**
* All tools are in /home/computeruse/tools/
* Before creating a new tool, explore existing ones: ls /home/computeruse/tools/
* Each tool has README.md explaining its purpose and how to use it
* Example: cat /home/computeruse/tools/pdf_converter/README.md

**CREATING NEW TOOLS:**
* Create in /home/computeruse/tools/your_tool_name/
* Use lowercase_with_underscores naming
* Must include README.md with: purpose, usage examples, inputs, outputs, requirements
* Make it reusable - other agents might use it

**TOOL RULES:**
1. NEVER modify or delete existing tools
2. If an existing tool almost works but needs changes, create a NEW tool
3. Use descriptive, specific names (pdf_to_text_converter, not converter)
4. Document your tool well in README.md
</TOOL_MANAGEMENT>

<FILE_DELIVERY>
**DELIVERING FILES TO THE USER:**

**CRITICAL: All completed work and deliverables MUST be uploaded to S3 for user download.**

Whenever you create files that the user needs to access, download, or that represent completed project deliverables, you MUST upload them to S3. This includes:
- Reports, documents, PDFs
- Websites, web applications
- Images, videos, media files
- Data files, spreadsheets, databases
- Code projects, applications
- Any packaged deliverables
- Any file the user explicitly requests
- Any completed project output

**Step 1: Package your deliverable**
```bash
# Navigate to your completed work
cd /home/computeruse/project/

# For a single file:
cp final_report.pdf /tmp/deliverable.pdf

# For multiple files/directories - ALWAYS ZIP THEM TOGETHER:
zip -r /tmp/deliverable.zip your_deliverable_folder/

# For websites/webapps:
tar -czf /tmp/website.tar.gz webapp/

# For code projects:
zip -r /tmp/project_name.zip project_folder/ -x "*/node_modules/*" "*/.git/*" "*/venv/*"
```

**Step 2: Upload to S3 (REQUIRED)**
```bash
# Upload your packaged file using environment variables
aws s3 cp /tmp/deliverable.zip \
  s3://""" + S3_BUCKET + """/outputs/""" + CHAT_ID + """/deliverable.zip --acl public-read

# Or for single files:
aws s3 cp /tmp/final_report.pdf \
  s3://""" + S3_BUCKET + """/outputs/""" + CHAT_ID + """/final_report.pdf --acl public-read
```

**Step 3: Provide the download URL to the user**
The download URL format is:
```
https://""" + S3_BUCKET + """.s3.amazonaws.com/outputs/""" + CHAT_ID + """/filename.ext
```

You can construct it in bash:
```bash
echo "Download your file here: https://""" + S3_BUCKET + """.s3.amazonaws.com/outputs/""" + CHAT_ID + """/deliverable.zip"
```

**ENVIRONMENT VARIABLES:**
* `$S3_BUCKET` - The S3 bucket name (automatically set)
* `$CHAT_ID` - The unique chat/session identifier (automatically set)
* These are pre-configured - just use them in your commands
* Verify they're set: `echo $S3_BUCKET` and `echo $CHAT_ID`

**AWS CREDENTIALS - ALREADY CONFIGURED:**
* AWS credentials are automatically configured via IRSA (IAM Roles for Service Accounts)
* Simply use the AWS CLI - authentication is handled automatically
* NEVER run `aws configure`
* NEVER create or modify AWS access keys or secret keys
* NEVER set AWS credential environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
* NEVER create or modify ~/.aws/ directories or config files
* Just use: `aws s3 cp <file> s3://$S3_BUCKET/outputs/$CHAT_ID/<filename> --acl public-read`

**WHEN TO UPLOAD TO S3:**
✓ User explicitly asks for a downloadable file
✓ User says they cannot access the VM
✓ Your task is complete and produces deliverable files
✓ You've created any final output: reports, websites, apps, images, videos, data files
✓ User needs to download or access the completed work
✓ Project is finished and ready for delivery
✓ ANY time you create files intended for the user

**PACKAGING BEST PRACTICES:**
- Use descriptive filenames: `marketing_campaign_q4.zip` not `output.zip`
- Include a README.md if the deliverable is complex or contains multiple files
- Test that all files are included before uploading
- For websites/apps, ensure all dependencies and assets are included
- Remove unnecessary files: node_modules, .git, venv, __pycache__, .DS_Store
- Verify the package can be extracted and used independently

**EXAMPLE WORKFLOW:**
```bash
# 1. Complete your work in /project/
cd /home/computeruse/project/

# 2. Package it appropriately
zip -r /tmp/website_final.zip webapp/ -x "*/node_modules/*"

# 3. Upload to S3 using environment variables
aws s3 cp /tmp/website_final.zip \
  s3://""" + S3_BUCKET + """/outputs/""" + CHAT_ID + """/website_final.zip --acl public-read

# 4. Construct and display the download URL
echo "Your website is ready! Download it here:"
echo "https://""" + S3_BUCKET + """.s3.amazonaws.com/outputs/""" + CHAT_ID + """/website_final.zip"
```

**MULTIPLE DELIVERABLES:**
If you have multiple separate deliverables (report + data + images), **ALWAYS combine them into a single ZIP file**:
```bash
# Create a temporary directory for all deliverables
mkdir -p /tmp/deliverables

# Copy all files to the directory
cp /home/computeruse/project/report.pdf /tmp/deliverables/
cp /home/computeruse/project/data.xlsx /tmp/deliverables/
cp -r /home/computeruse/project/images/ /tmp/deliverables/

# Optionally add a README explaining the contents
cat > /tmp/deliverables/README.txt << 'EOF'
PROJECT DELIVERABLES
===================
- report.pdf: Final project report
- data.xlsx: Analysis data and results
- images/: All project images and graphics
EOF

# Zip everything together
cd /tmp
zip -r project_deliverables.zip deliverables/

# Upload the single ZIP file using environment variables
aws s3 cp /tmp/project_deliverables.zip \
  s3://""" + S3_BUCKET + """/outputs/""" + CHAT_ID + """/project_deliverables.zip --acl public-read

# Display the download URL
echo "All deliverables are ready! Download here:"
echo "https://""" + S3_BUCKET + """.s3.amazonaws.com/outputs/""" + CHAT_ID + """/project_deliverables.zip"
```

Then provide the single download link to the user. This makes it easier for the user to download everything at once.
</FILE_DELIVERY>

<COLLABORATION>
**WORKING WITH YOUR TEAM:**

* **Shared workspace:** /home/computeruse/project/
* **Your private space:** /home/computeruse/work/
* **Team visibility:** Others see everything in /home/computeruse/project/, nothing in /home/computeruse/work/

**Best practices:**
- Check /home/computeruse/project/ to see what team members have already done
- Build upon their work rather than duplicating
- Organize clearly so others can understand and use your work
- Add README files or comments when helpful
- Don't overwrite others' work - communicate through file organization

**Example collaboration:**
Agent 1 creates: /home/computeruse/project/webapp/structure/
Agent 2 adds: /home/computeruse/project/webapp/backend/
Agent 3 adds: /home/computeruse/project/webapp/frontend/
Agent 4 polishes: /home/computeruse/project/webapp/ (final touches)
Final: Package /home/computeruse/project/webapp/ and upload to S3
</COLLABORATION>

<SYSTEM_ACCESS>
* You have full sudo access without requiring a password
* Use sudo for system-level operations: installing packages, fixing permissions, system configuration
* Example: sudo apt-get install <package>, sudo chown -R $USER:$USER <path>
* Make sure to use sudo responsibly and only when necessary
</SYSTEM_ACCESS>

<IMPORTANT_BEHAVIORAL_NOTES>
**Firefox Usage:**
* When Firefox shows a startup wizard, IGNORE it - don't click anything
* Click directly on the address bar where it says "Search or enter address"
* Type your URL or search term there and press Enter

**Reading PDFs:**
* If you need to read an entire PDF (not just view it):
```bash
  curl -o document.pdf https://example.com/document.pdf
  sudo apt-get install -y poppler-utils
  pdftotext document.pdf document.txt
  cat document.txt  # or use str_replace_based_edit_tool
```
* This is much faster and more accurate than taking many screenshots

**Text Input in Forms:**
* After entering text in any form field:
  1. Enter the text
  2. Wait 2 seconds
  3. Take a screenshot to verify text was entered correctly
  4. Check for: typos, extra spaces, missing characters
  5. Only proceed if the text is exactly correct
* This prevents form submission errors and data loss
</IMPORTANT_BEHAVIORAL_NOTES>"""

SYSTEM_PROMPT += f"""

<PROJECT_MEMORY>
You coordinate with other agents using TWO files in /home/computeruse/project/:

## 1. STATUS FILE: {PROJECT_STATUS_PATH}
**Purpose:** Current project state - what exists RIGHT NOW
**Operation:** OVERWRITE (not append)
**When to update:** When you START work and when you FINISH

**Format:**
```markdown
# Project Status
**Last Updated:** [YYYY-MM-DD HH:MM] by [Your Agent Name]
**State:** [Not Started | In Progress | Blocked | Complete]

## Current Deliverables
- [Component]: [Status] ([key files])

## Active Work
[What you're doing right now, or "None"]

## Blocked
[Any blockers, or "None"]

## Next Up
[What should happen next]
```

## 2. LOG FILE: {PROJECT_LOG_PATH}
**Purpose:** Historical record of completed work
**Operation:** APPEND ONLY (never modify previous entries)
**When to update:** When you COMPLETE a task

**Format:**
```markdown
## [Agent-Name] Task Description - YYYY-MM-DD HH:MM
**Delivered:** [1-2 sentence summary of what you built/did]
**Files:** [Full paths to key files created/modified]
**Decisions:** [Key choices you made and why]
**Handoff:** [What the next agent needs to know]

---
```

## YOUR WORKFLOW:
1. **ON START:** Read both files → Update STATUS (claim your work in "Active Work")
2. **DURING WORK:** Focus on your task (no logging needed)
3. **ON FINISH:** Update STATUS (new state, clear Active Work) → Append to LOG (your summary)

## CRITICAL RULES:
- ALWAYS update both files before you finish
- STATUS: Overwrite entire file with current state
- LOG: Only append, never modify existing entries
- Include FULL PATHS to files you create
- Explain your DECISIONS so future agents understand your choices
</PROJECT_MEMORY>
"""


# === COMPLETION PROMPT (used to force logging before exit) ===
LOG_COMPLETION_MESSAGE = """MANDATORY: Update project memory before finishing.

You must update BOTH files now:

**1. STATUS FILE ({status_path})** - OVERWRITE with current state:
```markdown
# Project Status
**Last Updated:** {timestamp} by Agent
**State:** [Complete or current state]

## Current Deliverables
[List what exists now with file paths]

## Active Work
None

## Blocked
None

## Next Up
[What should happen next, or "Project complete"]
```

**2. LOG FILE ({log_path})** - APPEND your entry:
```markdown
## [Agent] Task Description - {timestamp}
**Delivered:** [What you accomplished]
**Files:** [Key files with full paths]
**Decisions:** [Why you made key choices]
**Handoff:** [What next agent needs to know]

---
```

Use str_replace_editor to make these updates NOW."""

class ConversationStore:
    def __init__(self, pool: aiomysql.Pool):
        self.pool = pool

    @staticmethod
    async def create(
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        db: Optional[str] = None,
    ):
        """
        Create a ConversationStore instance with either provided parameters or environment variables.
        """
        connection_params = {
            "host": host or os.getenv("AGENT_DB_HOST", "localhost"),
            "port": port or int(os.getenv("AGENT_DB_PORT", "3306")),
            "user": user or os.getenv("AGENT_DB_USER", "root"),
            "password": password or os.getenv("AGENT_DB_PASSWORD", ""),
            "db": db or os.getenv("AGENT_DB_NAME", "multiai"),
            "autocommit": True,
        }

        try:
            pool = await aiomysql.create_pool(**connection_params)
            return ConversationStore(pool)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MySQL: {str(e)}") from e
            
    async def create_conversation(
        self,
        model: str,
        conv_type: str,
        status: str,
        # NEW parameters
        chat_id: Optional[int] = None,           # Laravel chat ID
        session_id: Optional[str] = None,        # K8s session ID
        parent_chat_id: Optional[int] = None,    # Parent computer_use_chats.id
        agent_name: Optional[str] = None,        # Agent identifier
    ) -> int:
        """Create a new conversation record."""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO computer_use_chats (
                        name, type, status, status_updated_at,
                        chat_id, session_id, parent_chat_id, agent_name,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        model,
                        conv_type,
                        status,
                        datetime.utcnow(),
                        chat_id,
                        session_id,
                        parent_chat_id,
                        agent_name,
                        datetime.utcnow(),
                        datetime.utcnow(),
                    ),
                )
                await conn.commit()
                return cur.lastrowid
            
    async def get_conversation(self, conversation_id: int) -> Optional[dict]:
        """Get a single conversation by ID."""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    """
                    SELECT 
                        id,
                        chat_id,
                        session_id,
                        agent_name,
                        parent_chat_id,
                        name as model,
                        type,
                        status,
                        status_message,
                        created_at,
                        completed_at
                    FROM computer_use_chats
                    WHERE id = %s
                    """,
                    (conversation_id,)
                )
                return await cur.fetchone()

    async def store_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        raw_content: str,
        tool_id: Optional[str] = None,
        is_error: bool = False,
        image_data: Optional[str] = None,
    ):
        """Store a message associated with a conversation"""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO computer_use_chat_messages (
                        computer_use_chat_id, role, content, tool_id,
                        is_error, timestamp, image_data, raw_content, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        conversation_id,
                        role,
                        content,
                        tool_id,
                        is_error,
                        datetime.utcnow(),
                        image_data,
                        raw_content,
                        datetime.utcnow(),
                        datetime.utcnow(),
                    ),
                )
                await conn.commit()

    async def mark_completed(self, conversation_id: int):
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE computer_use_chats
                    SET completed_at = %s
                    WHERE id = %s AND type = 'single'
                """,
                    (datetime.utcnow(), conversation_id),
                )
                await conn.commit()

    async def update_status(
        self,
        conversation_id: int,
        status: str,
        message: Optional[str] = None,
    ) -> None:
        """Update the status and related fields in the database."""
        now = datetime.utcnow()

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                query = """
                    UPDATE computer_use_chats
                    SET status = %s, status_updated_at = %s, status_message = %s
                """
                params: list = [status, now, message]

                if status == "completed":
                    query += ", completed_at = %s"
                    params.append(now)

                query += " WHERE id = %s"
                params.append(conversation_id)

                await cur.execute(query, tuple(params))
                await conn.commit()

    async def store_api_request(
        self,
        conversation_id: int,
        iteration: int,
        request_data: dict,
    ) -> int:
        """Store full API request for debugging."""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO computer_use_api_requests (
                        computer_use_chat_id, iteration, request_data, created_at
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (
                        conversation_id,
                        iteration,
                        json.dumps(request_data, default=str),
                        datetime.utcnow(),
                    ),
                )
                await conn.commit()
                return cur.lastrowid

    async def update_api_request_response(
        self,
        api_request_id: int,
        response_data: dict,
    ):
        """Update API request row with response data."""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE computer_use_api_requests 
                    SET response_data = %s, response_at = %s
                    WHERE id = %s
                    """,
                    (
                        json.dumps(response_data, default=str),
                        datetime.utcnow(),
                        api_request_id,
                    ),
                )
                await conn.commit()

    async def create_spawn(
        self,
        parent_conversation_id: int,
        chat_id: int,
        user_id: int,
        agent_id: str,
        agent_name: str,
        display_name: Optional[str],
        parent_agent_id: str,
        session_id: str,
        parent_session_id: str,
        system_prompt: str,
        task: str,
        wait_for_completion: bool = True,
        cleanup_on_complete: bool = True,
    ) -> int:
        """Create a spawn record for a sub-agent."""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO sub_agent_spawns (
                        parent_conversation_id,
                        chat_id,
                        user_id,
                        agent_id,
                        agent_name,
                        display_name,
                        parent_agent_id,
                        session_id,
                        parent_session_id,
                        system_prompt,
                        task,
                        wait_for_completion,
                        cleanup_on_complete,
                        status,
                        created_at,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    parent_conversation_id,
                    chat_id,
                    user_id,
                    agent_id,
                    agent_name,
                    display_name,
                    parent_agent_id,
                    session_id,
                    parent_session_id,
                    system_prompt,
                    task,
                    wait_for_completion,
                    cleanup_on_complete,
                    "spawning",
                    datetime.utcnow(),
                    datetime.utcnow(),
                ))
                await conn.commit()
                return cur.lastrowid
    
    async def update_spawn(
        self,
        spawn_id: int,
        child_conversation_id: Optional[int] = None,
        pod_name: Optional[str] = None,
        status: Optional[str] = None,
        result_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """Update a spawn record."""
        updates = []
        values = []
        
        if child_conversation_id is not None:
            updates.append("child_conversation_id = %s")
            values.append(child_conversation_id)
        
        if pod_name is not None:
            updates.append("pod_name = %s")
            values.append(pod_name)
        
        if status is not None:
            updates.append("status = %s")
            values.append(status)
        
        if result_summary is not None:
            updates.append("result_summary = %s")
            values.append(result_summary[:5000])  # Truncate if needed
        
        if error_message is not None:
            updates.append("error_message = %s")
            values.append(error_message[:2000])
        
        if started_at is not None:
            updates.append("started_at = %s")
            values.append(started_at)
        
        if completed_at is not None:
            updates.append("completed_at = %s")
            values.append(completed_at)
        
        if not updates:
            return
        
        updates.append("updated_at = %s")
        values.append(datetime.utcnow())
        values.append(spawn_id)
        
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                query = f"UPDATE sub_agent_spawns SET {', '.join(updates)} WHERE id = %s"
                await cur.execute(query, tuple(values))
                await conn.commit()
    
    async def get_spawn(self, spawn_id: int) -> Optional[dict]:
        """Get a spawn record."""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT * FROM sub_agent_spawns WHERE id = %s",
                    (spawn_id,)
                )
                return await cur.fetchone()
    
    async def get_spawns_for_chat(self, chat_id: int) -> list[dict]:
        """Get all spawns for a chat."""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT * FROM sub_agent_spawns WHERE chat_id = %s ORDER BY created_at ASC",
                    (chat_id,)
                )
                return await cur.fetchall()

def _update_status_file_on_exit(exit_status: str, agent_name: str = "Agent"):
    """Update the status file when agent exits abnormally."""
    _ensure_project_dir()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    
    try:
        # Read existing status or create new
        existing_content = ""
        if os.path.exists(PROJECT_STATUS_PATH):
            with open(PROJECT_STATUS_PATH, "r") as f:
                existing_content = f.read()
        
        # If agent didn't complete normally, add a warning to status
        if exit_status not in ("completed",):
            warning = f"\n\n> ⚠️ **System Note:** {agent_name} exited with status '{exit_status}' at {timestamp}. Work may be incomplete.\n"
            
            with open(PROJECT_STATUS_PATH, "a") as f:
                f.write(warning)
                
    except Exception as e:
        logger.error(f"Failed to update status file on exit: {e}")


def _append_system_log_entry(exit_status: str, agent_name: str = "Agent"):
    """Append a system entry to the log file when agent exits."""
    _ensure_project_dir()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    
    try:
        if exit_status == "completed":
            # Agent completed normally - it should have logged already
            # Just add a small system note
            log_entry = f"\n> [SYSTEM] {agent_name} completed successfully at {timestamp}\n"
        elif exit_status in ("cancelled", "paused", "stopping"):
            log_entry = f"""
## [SYSTEM] {agent_name} Interrupted - {timestamp}
**Status:** {exit_status}
**Note:** Agent was interrupted by user. Check status file for current state.

---
"""
        else:
            # Failed or unknown
            log_entry = f"""
## [SYSTEM] {agent_name} Stopped Unexpectedly - {timestamp}
**Status:** {exit_status}
**Note:** Agent did not complete normally. Review recent work and status file.

---
"""
        
        with open(PROJECT_LOG_PATH, "a") as f:
            f.write(log_entry)
            
    except Exception as e:
        logger.error(f"Failed to append system log entry: {e}")


from typing import Optional

class HeartbeatTracker:
    def __init__(self, store: ConversationStore, conv_id: int, min_interval: int = 30):
        self.store = store
        self.conv_id = conv_id
        self.min_interval = min_interval
        self.last_beat: float = 0
    
    async def beat(self, phase: Optional[str] = None) -> None:  # Fixed
        now = time.time()
        if now - self.last_beat >= self.min_interval:
            await self._update(phase)
            self.last_beat = now
    
    async def _update(self, phase: Optional[str] = None) -> None:  # Fixed
        try:
            async with self.store.pool.acquire() as conn:
                async with conn.cursor() as cur:
                    # Only update status_message if phase is provided
                    if phase:
                        await cur.execute(
                            """UPDATE computer_use_chats 
                               SET last_heartbeat_at = %s, 
                                   updated_at = %s,
                                   status_message = %s 
                               WHERE id = %s""",
                            (datetime.utcnow(), datetime.utcnow(), phase, self.conv_id)
                        )
                    else:
                        await cur.execute(
                            """UPDATE computer_use_chats 
                               SET last_heartbeat_at = %s, 
                                   updated_at = %s
                               WHERE id = %s""",
                            (datetime.utcnow(), datetime.utcnow(), self.conv_id)
                        )
                    await conn.commit()
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")

async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 32768,
    conversation_store: Optional[ConversationStore] = None,
    current_conversation_id: Optional[int] = None,
    conversation_type: str = "continuous",
    tool_version: ToolVersion,
    thinking_budget: int | None = None,
    token_efficient_tools_beta: bool = False,
    use_extended_context: bool = False,
    agent_name: str = "Agent", 
    cleanup_on_complete: bool = True,
    session_id: Optional[str] = None,
    spawn_id: Optional[int] = None,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    logger.info(f"[Loop] Starting sampling_loop for conversation {current_conversation_id}")
    logger.info(f"[Loop] Model: {model}, Provider: {provider}")
    logger.debug(f"[Loop] Tool version: {tool_version}")
    logger.info(f"[Loop] cleanup_on_complete={cleanup_on_complete}, session_id={session_id}")

    # Get broker info for self-cleanup
    broker_url = os.getenv("BROKER_URL", "http://broker:8001")
    broker_token = os.getenv("BROKER_TOKEN", "")

    _ensure_project_dir()

    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    subagent_tool = SubAgentTool()
    tool_collection = ToolCollection(
        *(ToolCls() for ToolCls in tool_group.tools),
        subagent_tool,  # Include it during initialization
        CredentialsTool()  # Add CredentialsTool to the collection
    )
    logger.debug(f"[Loop] Tools loaded: {[t.name for t in tool_collection.tools]}")
    
    project_memory = _load_project_memory()
    deployment_info = _get_deployment_capabilities()
    full_system_prompt = f"{SYSTEM_PROMPT}\n{project_memory}\n{deployment_info}\n{system_prompt_suffix}"

    system = BetaTextBlockParam(
        type="text",
        text=full_system_prompt,
    )

    # Create conversation store if not provided
    if conversation_store is None:
        # try:
        logger.debug(f"[Loop] Creating new ConversationStore")
        conversation_store = await ConversationStore.create()
    # except ConnectionError as e:
    #     return messages, None

    if current_conversation_id is None:
        logger.debug(f"[Loop] Creating new conversation")
        current_conversation_id = await conversation_store.create_conversation(
            model=model,
            conv_type=conversation_type,
            status="running",
            agent_name=agent_name,
        )

        logger.info(f"[Loop] Created conversation: {current_conversation_id}")

    heartbeat_tracker = HeartbeatTracker(conversation_store, current_conversation_id)
    await heartbeat_tracker.beat("starting")

    # Set the conversation ID on SubAgentTool so it can track hierarchy
    subagent_tool.my_conversation_id = current_conversation_id
    subagent_tool.conversation_store = conversation_store
    logger.debug(f"[Loop] Set subagent_tool.my_conversation_id = {current_conversation_id}")

    iteration = 0
    prompted_for_log = False  # Track if we've asked agent to log
    final_status = "running"  # Track exit status for finally block

    try:
        while True:
            iteration += 1
            logger.info(f"[Loop] === Iteration {iteration} ===")

            # Check for pause/stop signals at the start of each iteration
            try:
                async with conversation_store.pool.acquire() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "SELECT status FROM computer_use_chats WHERE id = %s",
                            (current_conversation_id,),
                        )
                        result = await cur.fetchone()
                        current_status = result[0] if result else "failed"
                        logger.debug(f"[Loop] Current status from DB: {current_status}")

                if current_status == "stopping":
                    logger.info(f"[Loop] Received stop signal, exiting")
                    final_status = "cancelled"
                    await conversation_store.update_status(
                        current_conversation_id,
                        "cancelled",
                        "Task cancelled by user.",
                    )
                    break  # Exit the loop gracefully

                if current_status == "pausing":
                    logger.info(f"[Loop] Received pause signal, exiting")
                    final_status = "paused"
                    await conversation_store.update_status(
                        current_conversation_id,
                        "paused",
                        "Task paused by user.",
                    )
                    break  # Exit the loop gracefully

            except Exception as e:
                logger.exception(f"[Loop] Error checking status: {e}")
                final_status = "failed"
                await conversation_store.update_status(
                    current_conversation_id,
                    "failed",
                    f"Internal error during status check: {e}",
                )
                break

            enable_prompt_caching = False
            betas = [tool_group.beta_flag] if tool_group.beta_flag else []
            if token_efficient_tools_beta:
                betas.append("token-efficient-tools-2025-02-19")
            if use_extended_context:
                betas.append("context-1m-2025-08-07")
            image_truncation_threshold = only_n_most_recent_images or 0

            if provider == APIProvider.ANTHROPIC:
                client = Anthropic(api_key=api_key, max_retries=4, timeout=httpx.Timeout(300.0, connect=10.0))
                enable_prompt_caching = True
            elif provider == APIProvider.VERTEX:
                client = AnthropicVertex()
            elif provider == APIProvider.BEDROCK:
                client = AnthropicBedrock()

            if enable_prompt_caching:
                betas.append(PROMPT_CACHING_BETA_FLAG)
                _inject_prompt_caching(messages)
                # Because cached reads are 10% of the price, we don't think it's
                # ever sensible to break the cache by truncating images
                only_n_most_recent_images = 90
                # Use type ignore to bypass TypedDict check until SDK types are updated
                system["cache_control"] = {"type": "ephemeral"}  # type: ignore

            if only_n_most_recent_images:
                _maybe_filter_to_n_most_recent_images(
                    messages,
                    only_n_most_recent_images,
                    min_removal_threshold=image_truncation_threshold,
                )
            extra_body = {}
            # if thinking_budget:
            #     # Ensure we only send the required fields for thinking
            #     extra_body = {
            #         "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
            #     }

            # Store the last user message into db
            last_user_message = messages[-1]
            if last_user_message["role"] == "user":
                if current_conversation_id is not None:

                    logger.debug(f"[Loop] Storing user message to DB")
                    try:
                        await conversation_store.store_message(
                            conversation_id=current_conversation_id,
                            role="user",
                            content=json.dumps(last_user_message["content"]),
                            raw_content=json.dumps(last_user_message),
                            tool_id="user-input",
                        )
                        logger.debug(f"[Loop] User message stored successfully")
                    except Exception as e:
                        logger.exception(f"[Loop] Failed to store user message: {e}")

            api_request_id: Optional[int] = None

            try:
                request_data = {
                    "model": model,
                    "system_prompt": full_system_prompt,
                    "system_prompt_length": len(full_system_prompt),
                    "messages": messages,
                    "messages_count": len(messages),
                    "tools": tool_collection.to_params(),
                    "betas": betas,
                    "extra_body": extra_body,
                    "max_tokens": max_tokens,
                }
                api_request_id = await conversation_store.store_api_request(
                    conversation_id=current_conversation_id,
                    iteration=iteration,
                    request_data=request_data,
                )
                logger.debug(f"[Loop] API request logged for iteration {iteration}")
            except Exception as e:
                logger.warning(f"[Loop] Failed to log API request: {e}")

            # Call the API
            logger.info(f"[Loop] Calling Claude API...")
            logger.debug(f"[Loop] Messages count: {len(messages)}")

            # we use raw_response to provide debug information to streamlit. Your
            # implementation may be able call the SDK directly with:
            # `response = client.messages.create(...)` instead.
            try:

                await heartbeat_tracker.beat("calling_api")

                raw_response = client.beta.messages.with_raw_response.create(
                    max_tokens=max_tokens if max_tokens is not None else 4096,
                    messages=messages,
                    model=model,
                    system=[system],
                    tools=tool_collection.to_params(),
                    betas=betas,
                    extra_body=extra_body,
                )
                # except (APIStatusError, APIResponseValidationError) as e:
                #     api_response_callback(e.request, e.response, e)
                #     return messages, current_conversation_id
                # except APIError as e:
                #     api_response_callback(e.request, e.body, e)
                #     return messages, current_conversation_id

                logger.info(f"[Loop] API call successful")

                api_response_callback(
                    raw_response.http_response.request, raw_response.http_response, None
                )

                await heartbeat_tracker.beat("processing_response")

                response = raw_response.parse()

                try:
                    response_data = {
                        "id": response.id,
                        "model": response.model,
                        "stop_reason": response.stop_reason,
                        "stop_sequence": response.stop_sequence,
                        "content": [block.model_dump() for block in response.content],
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                        } if response.usage else None,
                    }
                    await conversation_store.update_api_request_response(
                        api_request_id=api_request_id,
                        response_data=response_data,
                    )
                    logger.debug(f"[Loop] API response logged for iteration {iteration}")
                except Exception as e:
                    logger.warning(f"[Loop] Failed to log API response: {e}")

                logger.debug(f"[Loop] Response stop_reason: {response.stop_reason}")
                logger.debug(f"[Loop] Response content blocks: {len(response.content)}")

                response_params = _response_to_params(response)
                response_message: BetaMessageParam = {
                    "role": "assistant",
                    "content": response_params,
                }
                messages.append(response_message)

                if current_conversation_id is not None:
                    logger.debug(f"[Loop] Storing assistant message to DB")
                    try:
                        await conversation_store.store_message(
                            conversation_id=current_conversation_id,
                            role="assistant",
                            content=json.dumps(response_params),
                            tool_id="response",
                            raw_content=json.dumps(response_message),
                        )
                        logger.debug(f"[Loop] Assistant message stored successfully")
                    except Exception as e:
                        logger.exception(f"[Loop] Failed to store assistant message: {e}")

                tool_result_content: list[BetaToolResultBlockParam] = []
                for content_block in response_params:
                    output_callback(content_block)

                    if (
                        isinstance(content_block, dict)
                        and content_block.get("type") == "tool_use"
                    ):
                        # Type narrowing for tool use blocks
                        tool_use_block = cast(BetaToolUseBlockParam, content_block)
                        tool_name = tool_use_block.get("name", "unknown")
                        tool_input = cast(dict[str, Any], tool_use_block.get("input", {}))

                        await heartbeat_tracker.beat(f"executing_tool:{tool_name}")

                        logger.info(f"[Loop] Executing tool: {tool_use_block['name']}")
                        logger.debug(f"[Loop] Tool input: {tool_use_block.get('input', {})}")

                        try:
                            result = await tool_collection.run(
                                name=tool_use_block["name"],
                                tool_input=tool_input,
                            )

                            # --- NEW INTERRUPTION CHECK ---
                            # If the tool realized we are cancelled, it returns this string.
                            # We break the loop so the `finally` block runs and saves the log.
                            if result.output == "CANCELLED_BY_USER":
                                logger.info("[Loop] Tool detected cancellation signal. Exiting.")
                                # We set status so the finally block logs it correctly
                                final_status = "cancelled"
                                break 
                            # ------------------------------
                        except Exception as tool_error:
                            # Don't crash the loop - return error to Claude so it can retry
                            logger.error(f"[Loop] Tool execution error: {tool_error}")
                            result = ToolResult(error=f"Tool execution failed: {str(tool_error)}")
                        
                        tool_result_content.append(
                            _make_api_tool_result(result, tool_use_block["id"])
                        )
                        tool_output_callback(result, tool_use_block["id"])

                await heartbeat_tracker.beat("iteration_complete")

                if not tool_result_content:
                    logger.info(f"[Loop] No tool calls, task complete!")

                    if not prompted_for_log:
                        # First time seeing no tool calls - prompt agent to log
                        prompted_for_log = True
                        logger.info("[Loop] Prompting agent to update project memory before completion")
                        
                        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
                        log_prompt_text = LOG_COMPLETION_MESSAGE.format(
                            status_path=PROJECT_STATUS_PATH,
                            log_path=PROJECT_LOG_PATH,
                            timestamp=timestamp,
                        )
                        
                        log_prompt: BetaMessageParam = {
                            "role": "user",
                            "content": [{"type": "text", "text": log_prompt_text}]
                        }
                        messages.append(log_prompt)

                        if current_conversation_id is not None:
                            try:
                                await conversation_store.store_message(
                                    conversation_id=current_conversation_id,
                                    role="user",
                                    content=json.dumps(log_prompt["content"]),
                                    raw_content=json.dumps(log_prompt),
                                    tool_id="system-log-prompt",
                                )
                            except Exception as e:
                                logger.exception(f"[Loop] Failed to store log prompt: {e}")
                        
                        # Continue to next iteration to let agent respond
                        continue
                    
                    # Already prompted - agent is truly done
                    logger.info(f"[Loop] No tool calls after log prompt, task complete!")
                    final_status = "completed"

                    if current_conversation_id is not None:
                        await conversation_store.update_status(
                            current_conversation_id,
                            "completed",
                            "Task finished successfully without further tool use.",
                        )
                        await conversation_store.mark_completed(current_conversation_id)
                    
                    return messages, current_conversation_id

                result_message: BetaMessageParam = {
                    "role": "user",
                    "content": tool_result_content,
                }

                if current_conversation_id is not None:
                    logger.debug(f"[Loop] Storing tool results to DB")
                    try:
                        await conversation_store.store_message(
                            conversation_id=current_conversation_id,
                            role="tool",
                            content=json.dumps(tool_result_content),
                            tool_id="response",
                            raw_content=json.dumps(result_message),
                        )
                    except Exception as e:
                        logger.exception(f"[Loop] Failed to store tool results: {e}")

                messages.append(result_message)
                logger.debug(f"[Loop] Continuing to next iteration...")

            except (APIStatusError, APIResponseValidationError) as e:
                error_message = f"API Error: {e}"
                logger.exception(f"[Loop] {error_message}")
                await conversation_store.update_status(
                    current_conversation_id, "failed", error_message
                )
                api_response_callback(e.request, e.response, e)
                return messages, current_conversation_id
            except APIError as e:
                error_message = f"API Error: {e}"
                logger.exception(f"[Loop] {error_message}")
                await conversation_store.update_status(
                    current_conversation_id, "failed", error_message
                )
                api_response_callback(e.request, e.body, e)
                return messages, current_conversation_id
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                logger.exception(f"[Loop] {error_message}")
                try:
                    # Update status
                    await conversation_store.update_status(
                        current_conversation_id, "failed", error_message
                    )
                    
                    # Store the error as a message so user can see what happened
                    await conversation_store.store_message(
                        conversation_id=current_conversation_id,
                        role="system",
                        content=json.dumps([{"type": "error", "text": error_message}]),
                        raw_content=json.dumps({"role": "system", "content": error_message}),
                        tool_id="system-error",
                        is_error=True,
                    )
                except Exception as status_error:
                    logger.error(f"[Loop] Failed to update status/store error: {status_error}")
                
                return messages, current_conversation_id
    finally:
        # === ALWAYS UPDATE PROJECT MEMORY ON EXIT ===
        logger.info(f"[Loop] Finally block - exit status: {final_status}")
        
        # Update status file if abnormal exit
        _update_status_file_on_exit(final_status, agent_name)
        
        # Append to log file
        _append_system_log_entry(final_status, agent_name)

        # ═══════════════════════════════════════════════════════════════
        # Update spawn record if this is a sub-agent
        # ═══════════════════════════════════════════════════════════════
        if spawn_id and conversation_store:
            try:
                # Get the final response to use as result_summary
                final_summary = None
                if messages:
                    # Get the last assistant message as summary
                    for msg in reversed(messages):
                        if msg.get("role") == "assistant":
                            content = msg.get("content", [])
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        final_summary = block.get("text", "")[:5000]
                                        break
                            break
                
                # Map final_status to spawn status
                spawn_status = {
                    "completed": "completed",
                    "cancelled": "cancelled",
                    "paused": "cancelled",
                    "failed": "failed",
                }.get(final_status, "failed")
                
                await conversation_store.update_spawn(
                    spawn_id,
                    status=spawn_status,
                    result_summary=final_summary,
                    completed_at=datetime.utcnow(),
                )
                logger.info(f"[Loop] Updated spawn record {spawn_id} to {spawn_status}")
            except Exception as e:
                logger.error(f"[Loop] Failed to update spawn record: {e}")

        if cleanup_on_complete and session_id:
            logger.info(f"[Loop] Self-cleanup enabled, deleting session {session_id}")
            await _self_cleanup(broker_url, broker_token, session_id, final_status)
        else:
            logger.info(f"[Loop] Self-cleanup disabled or no session_id, pod will stay running")

async def _self_cleanup(broker_url: str, broker_token: str, session_id: str, reason: str):
    """
    Sub-agent cleans up its own pod by calling the broker.
    This is fire-and-forget - we're about to terminate anyway.
    """
    try:
        logger.info(f"[Loop] Requesting self-cleanup: {session_id}")
        
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.delete(
                f"{broker_url}/sessions/{session_id}",
                headers={"X-Broker-Token": broker_token}
            )
            
            if response.is_success:
                logger.info(f"[Loop] Self-cleanup successful")
            else:
                logger.warning(f"[Loop] Self-cleanup failed: {response.status_code} - {response.text}")
                
    except Exception as e:
        # Don't crash - we're cleaning up anyway
        logger.warning(f"[Loop] Self-cleanup error (non-fatal): {e}")

def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _response_to_params(
    response: BetaMessage,
) -> list[BetaContentBlockParam]:
    res: list[BetaContentBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            if block.text:
                res.append(BetaTextBlockParam(type="text", text=block.text))
            elif getattr(block, "type", None) == "thinking":
                # Handle thinking blocks - include signature field
                thinking_block = {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", None),
                }
                if hasattr(block, "signature"):
                    thinking_block["signature"] = getattr(block, "signature", None)
                res.append(cast(BetaContentBlockParam, thinking_block))
        else:
            # Handle tool use blocks normally
            res.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return res


def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(
            content := message["content"], list
        ):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                # Use type ignore to bypass TypedDict check until SDK types are updated
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(  # type: ignore
                    {"type": "ephemeral"}
                )
            else:
                if isinstance(content[-1], dict) and "cache_control" in content[-1]:
                    del content[-1]["cache_control"]  # type: ignore
                # we'll only every have one extra turn per loop
                break


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
