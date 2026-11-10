#!/usr/bin/env bash
#
# One time setup for the DagsHub remote. Reads .env so the token never ends
# up in your shell history or in a committed file.
#
#   bash scripts/setup_dagshub.sh
#
# On Windows run this from Git Bash, or just run the four dvc commands by
# hand from PowerShell.

set -e

if [ ! -f .env ]; then
  echo ".env not found. copy .env.example to .env and fill it in first."
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

if [ -z "$DAGSHUB_USER" ] || [ -z "$DAGSHUB_TOKEN" ]; then
  echo "DAGSHUB_USER and DAGSHUB_TOKEN must be set in .env"
  exit 1
fi

REMOTE_URL="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.s3"

if [ ! -d .dvc ]; then
  dvc init
fi

dvc remote add -d --force origin "$REMOTE_URL"
dvc remote modify origin endpointurl "https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.s3"
dvc remote modify --local origin access_key_id "$DAGSHUB_TOKEN"
dvc remote modify --local origin secret_access_key "$DAGSHUB_TOKEN"

echo
echo "dvc remote configured:"
dvc remote list
echo
echo "mlflow will log to: ${MLFLOW_TRACKING_URI}"
echo "now run: dvc repro split && dvc push"
