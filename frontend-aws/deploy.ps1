# deploy.ps1 - Deploy Political Strategy Frontend to AWS S3 + CloudFront
# 
# Usage:
#   .\deploy.ps1 -BackendApiUrl "https://xxx.execute-api.region.amazonaws.com" -BackendWsUrl "wss://xxx.execute-api.region.amazonaws.com/production"
#
# Parameters:
#   -Environment: Deployment environment (development, staging, production)
#   -StackName: CloudFormation stack name
#   -BackendApiUrl: Backend HTTP API URL
#   -BackendWsUrl: Backend WebSocket URL
#   -SkipBuild: Skip npm build step
#   -SkipInfra: Skip infrastructure deployment
#   -Region: AWS region (default: ap-south-1)

param(
    [string]$Environment = "production",
    [string]$StackName = "political-strategy-frontend",
    [Parameter(Mandatory=$true)]
    [string]$BackendApiUrl,
    [Parameter(Mandatory=$true)]
    [string]$BackendWsUrl,
    [switch]$SkipBuild,
    [switch]$SkipInfra,
    [string]$Region = "ap-south-1"
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Political Strategy Frontend Deploy" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Environment:    $Environment" -ForegroundColor Yellow
Write-Host "Stack Name:     $StackName-$Environment" -ForegroundColor Yellow
Write-Host "Region:         $Region" -ForegroundColor Yellow
Write-Host "Backend API:    $BackendApiUrl" -ForegroundColor Yellow
Write-Host "Backend WS:     $BackendWsUrl" -ForegroundColor Yellow
Write-Host ""

# Check prerequisites
Write-Host "[1/6] Checking prerequisites..." -ForegroundColor Green

# Check AWS CLI
try {
    $awsVersion = aws --version 2>&1
    Write-Host "  AWS CLI: $awsVersion" -ForegroundColor Gray
} catch {
    Write-Host "ERROR: AWS CLI not found. Please install: https://aws.amazon.com/cli/" -ForegroundColor Red
    exit 1
}

# Check SAM CLI
try {
    $samVersion = sam --version 2>&1
    Write-Host "  SAM CLI: $samVersion" -ForegroundColor Gray
} catch {
    Write-Host "ERROR: SAM CLI not found. Please install: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  Node.js: $nodeVersion" -ForegroundColor Gray
} catch {
    Write-Host "ERROR: Node.js not found. Please install: https://nodejs.org/" -ForegroundColor Red
    exit 1
}

# Step 2: Build the frontend
if (-not $SkipBuild) {
    Write-Host ""
    Write-Host "[2/6] Building frontend application..." -ForegroundColor Green
    
    # Create .env.production for build
    $envContent = @"
VITE_API_URL=$BackendApiUrl
VITE_WS_URL=$BackendWsUrl
VITE_ENVIRONMENT=$Environment
"@
    
    Set-Content -Path ".env.production" -Value $envContent -Encoding UTF8
    Write-Host "  Created .env.production" -ForegroundColor Gray
    
    # Install dependencies
    Write-Host "  Installing dependencies..." -ForegroundColor Gray
    npm install --silent
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: npm install failed" -ForegroundColor Red
        exit 1
    }
    
    # Build
    Write-Host "  Building production bundle..." -ForegroundColor Gray
    npm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Build failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "  Build completed successfully!" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "[2/6] Skipping build (--SkipBuild)" -ForegroundColor Yellow
}

# Step 3: Deploy infrastructure
if (-not $SkipInfra) {
    Write-Host ""
    Write-Host "[3/6] Deploying AWS infrastructure..." -ForegroundColor Green
    
    # Validate template first
    Write-Host "  Validating CloudFormation template..." -ForegroundColor Gray
    sam validate --template infrastructure/template.yaml --region $Region
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Template validation failed" -ForegroundColor Red
        exit 1
    }
    
    # Deploy stack
    Write-Host "  Deploying CloudFormation stack..." -ForegroundColor Gray
    sam deploy `
        --template-file infrastructure/template.yaml `
        --stack-name "$StackName-$Environment" `
        --capabilities CAPABILITY_IAM `
        --region $Region `
        --parameter-overrides `
            "Environment=$Environment" `
            "BackendApiUrl=$BackendApiUrl" `
            "BackendWsUrl=$BackendWsUrl" `
        --no-confirm-changeset `
        --no-fail-on-empty-changeset
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Infrastructure deployment failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "  Infrastructure deployed successfully!" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "[3/6] Skipping infrastructure deployment (--SkipInfra)" -ForegroundColor Yellow
}

# Step 4: Get stack outputs
Write-Host ""
Write-Host "[4/6] Getting deployment information..." -ForegroundColor Green

$stackOutputsJson = aws cloudformation describe-stacks `
    --stack-name "$StackName-$Environment" `
    --region $Region `
    --query "Stacks[0].Outputs" `
    --output json 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to get stack outputs. Stack might not exist." -ForegroundColor Red
    Write-Host "Run without -SkipInfra to create the stack first." -ForegroundColor Yellow
    exit 1
}

$stackOutputs = $stackOutputsJson | ConvertFrom-Json

$bucketName = ($stackOutputs | Where-Object { $_.OutputKey -eq "FrontendBucketName" }).OutputValue
$distributionId = ($stackOutputs | Where-Object { $_.OutputKey -eq "CloudFrontDistributionId" }).OutputValue
$websiteUrl = ($stackOutputs | Where-Object { $_.OutputKey -eq "WebsiteUrl" }).OutputValue
$dashboardUrl = ($stackOutputs | Where-Object { $_.OutputKey -eq "DashboardUrl" }).OutputValue

Write-Host "  S3 Bucket:     $bucketName" -ForegroundColor Gray
Write-Host "  Distribution:  $distributionId" -ForegroundColor Gray
Write-Host "  Website URL:   $websiteUrl" -ForegroundColor Gray

# Step 5: Upload to S3
Write-Host ""
Write-Host "[5/6] Uploading to S3..." -ForegroundColor Green

# Check if dist folder exists
if (-not (Test-Path "dist")) {
    Write-Host "ERROR: dist folder not found. Run build first." -ForegroundColor Red
    exit 1
}

# Sync static assets with long cache
Write-Host "  Uploading static assets (cached)..." -ForegroundColor Gray
aws s3 sync dist/ "s3://$bucketName/" `
    --delete `
    --cache-control "max-age=31536000,public,immutable" `
    --exclude "*.html" `
    --exclude "*.json" `
    --region $Region

# Upload HTML and JSON with no-cache
Write-Host "  Uploading HTML/JSON (no-cache)..." -ForegroundColor Gray
aws s3 cp dist/index.html "s3://$bucketName/index.html" `
    --cache-control "no-cache,no-store,must-revalidate" `
    --content-type "text/html" `
    --region $Region

# Upload any JSON files (like manifest)
Get-ChildItem -Path dist -Filter "*.json" -Recurse | ForEach-Object {
    $relativePath = $_.FullName.Replace((Get-Location).Path + "\dist\", "").Replace("\", "/")
    aws s3 cp $_.FullName "s3://$bucketName/$relativePath" `
        --cache-control "no-cache,no-store,must-revalidate" `
        --content-type "application/json" `
        --region $Region
}

Write-Host "  Upload completed!" -ForegroundColor Gray

# Step 6: Invalidate CloudFront cache
Write-Host ""
Write-Host "[6/6] Invalidating CloudFront cache..." -ForegroundColor Green

$invalidationResult = aws cloudfront create-invalidation `
    --distribution-id $distributionId `
    --paths "/*" `
    --region us-east-1 `
    --output json | ConvertFrom-Json

$invalidationId = $invalidationResult.Invalidation.Id
Write-Host "  Invalidation ID: $invalidationId" -ForegroundColor Gray
Write-Host "  Cache invalidation in progress..." -ForegroundColor Gray

# Done!
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Deployment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Website URL:      $websiteUrl" -ForegroundColor Cyan
Write-Host "CloudWatch:       $dashboardUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: CloudFront propagation may take 5-10 minutes." -ForegroundColor Yellow
Write-Host ""

