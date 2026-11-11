#!/usr/bin/env bash
#
# One time setup for the DagsHub remote. Reads .env so the token never ends
# up in your shell history or in a committed file.
#
#   bash scripts/setup_dagshub.sh
#
# On Windows run this from Git Bash, or run the five dvc commands by hand
# from PowerShell.
#
# Note the url is s3://dvc, not the https address. DagsHub exposes an S3
# compatible endpoint, and the bucket inside it is called dvc. Giving dvc an
# https:// url makes it treat the remote as a plain http remote, which then
# rejects endpointurl and the credentials with "extra keys not allowed".

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

ENDPOINT="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.s3"

if [ ! -d .dvc ]; then
  dvc init
fi

# remove any earlier attempt so this script can be run again safely
dvc remote remove origin 2>/dev/null || true

dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl "$ENDPOINT"
dvc remote modify --local origin access_key_id "$DAGSHUB_TOKEN"
dvc remote modify --local origin secret_access_key "$DAGSHUB_TOKEN"

echo
echo "dvc remote configured:"
dvc remote list
echo
echo "the token lives in .dvc/config.local, which git ignores"
echo "mlflow will log to: ${MLFLOW_TRACKING_URI}"
echo
echo "now run: dvc repro split"
