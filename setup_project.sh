#!/bin/bash
set -e

# Generate a unique project ID (e.g., ml-pipeline-a1b2c3)
RANDOM_SUFFIX=$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 6)
NEW_PROJECT_ID="ml-pipeline-$RANDOM_SUFFIX"

echo "🚀 Creating new GCP Project: $NEW_PROJECT_ID..."
gcloud projects create $NEW_PROJECT_ID --name="ML Pipeline Project"

echo "🔄 Setting active project..."
gcloud config set project $NEW_PROJECT_ID

echo "💰 Fetching available billing accounts..."
gcloud beta billing accounts list

echo "---------------------------------------------------"
echo "Please enter your Billing Account ID from the list above (Format: XXXXXX-XXXXXX-XXXXXX):"
read BILLING_ACCOUNT_ID

echo "🔗 Linking billing account to project..."
gcloud beta billing projects link $NEW_PROJECT_ID --billing-account $BILLING_ACCOUNT_ID

echo "⚙️  Enabling Cloud Storage API..."
gcloud services enable storage.googleapis.com

echo "✅ Project setup complete! Your new Project ID is: $NEW_PROJECT_ID"
