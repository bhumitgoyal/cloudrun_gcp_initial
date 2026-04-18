#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_dev.sh  —  GoHappy Club Bot · Local Development Launcher
#
# Uses cloudflared (Cloudflare Quick Tunnel) instead of ngrok.
# No account or auth token needed for basic tunneling.
#
# Usage:
#   chmod +x run_dev.sh && ./run_dev.sh
#
# Prerequisites:
#   brew install cloudflared
#   pip install -r requirements.txt   (done automatically below)
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
BOLD="\033[1m"; GREEN="\033[92m"; YELLOW="\033[93m"
CYAN="\033[96m"; RED="\033[91m"; END="\033[0m"

banner() {
  echo -e "\n${BOLD}══════════════════════════════════════════════════${END}"
  echo -e "  $1"
  echo -e "${BOLD}══════════════════════════════════════════════════${END}\n"
}
die() { echo -e "\n  ${RED}${BOLD}FATAL: $1${END}\n"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load .env ────────────────────────────────────────────────────────────────
ENV_FILE="$SCRIPT_DIR/.env"
banner "${CYAN}Loading .env${END}"
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
  echo -e "  ${GREEN}✅ Environment variables loaded${END}"
else
  echo -e "  ${YELLOW}⚠️  No .env file found — using existing env${END}"
fi
PORT="${PORT:-8080}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
banner "${CYAN}Pre-flight Checks${END}"

# Python — prefer conda base env which has all packages
CONDA_PYTHON=""
if command -v conda &>/dev/null; then
  CONDA_PYTHON=$(conda run -n base which python 2>/dev/null || true)
fi
PYTHON="${CONDA_PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)}"
[[ -z "$PYTHON" ]] && die "Python not found"
echo -e "  ${GREEN}✅ Python: $PYTHON ($("$PYTHON" --version 2>&1))${END}"

# cloudflared
if ! command -v cloudflared &>/dev/null; then
  die "cloudflared not found.\n     Install: brew install cloudflared"
fi
echo -e "  ${GREEN}✅ cloudflared: $(cloudflared --version 2>&1 | head -1)${END}"

# Required env vars
missing=0
for var in WHATSAPP_ACCESS_TOKEN WHATSAPP_PHONE_NUMBER_ID WHATSAPP_VERIFY_TOKEN; do
  val="${!var:-}"
  if [[ -z "$val" || "$val" == "<"*">" ]]; then
    echo -e "  ${RED}❌ $var is not set or has a placeholder value${END}"
    missing=1
  else
    echo -e "  ${GREEN}✅ $var = ${val:0:8}...${val: -6}${END}"
  fi
done
[[ $missing -eq 1 ]] && die "Fix missing variables in .env before continuing."

# ── Install Python dependencies ───────────────────────────────────────────────
banner "${CYAN}Installing Python Dependencies${END}"
if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
  echo -e "  Running: pip install -r requirements.txt\n"
  "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt" \
    --quiet --disable-pip-version-check
  echo -e "\n  ${GREEN}✅ Dependencies installed${END}"
else
  echo -e "  ${YELLOW}⚠️  requirements.txt not found${END}"
fi

# Verify uvicorn
"$PYTHON" -c "import uvicorn" 2>/dev/null || {
  echo -e "  Installing uvicorn…"
  "$PYTHON" -m pip install "uvicorn[standard]" --quiet
}
echo -e "  ${GREEN}✅ uvicorn ready${END}"

# ── Cleanup on exit ───────────────────────────────────────────────────────────
APP_PID=""; TUNNEL_PID=""
cleanup() {
  echo -e "\n${YELLOW}Shutting down…${END}"
  [[ -n "$APP_PID"    ]] && kill "$APP_PID"    2>/dev/null && echo "  Stopped FastAPI      (pid $APP_PID)"
  [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null && echo "  Stopped cloudflared  (pid $TUNNEL_PID)"
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ── Start FastAPI server ──────────────────────────────────────────────────────
banner "${CYAN}Starting FastAPI Server${END}"
echo -e "  Port: ${PORT}"
"$PYTHON" "$SCRIPT_DIR/main.py" > /tmp/gohappy_server.log 2>&1 &
APP_PID=$!

echo -ne "  Waiting for server"
SERVER_UP=0
for i in $(seq 1 20); do
  sleep 0.5
  HEALTH=$(curl -sf "http://localhost:${PORT}/health" 2>/dev/null || true)
  if [[ -n "$HEALTH" ]]; then SERVER_UP=1; break; fi
  echo -ne "."
done
echo ""

if [[ $SERVER_UP -eq 0 ]]; then
  echo -e "  ${RED}❌ Server failed to start. Last log output:${END}"
  echo "──────────────────────────────────────────"
  tail -30 /tmp/gohappy_server.log
  echo "──────────────────────────────────────────"
  exit 1
fi
echo -e "  ${GREEN}✅ Server is up (pid $APP_PID)${END}"
echo -e "  ${CYAN}Health: ${HEALTH}${END}"

# ── Start Cloudflare Quick Tunnel ─────────────────────────────────────────────
banner "${CYAN}Starting Cloudflare Quick Tunnel${END}"
TUNNEL_LOG="/tmp/cloudflared_dev.log"

# cloudflared quick-tunnel — URL is printed to stderr, so we capture both streams
cloudflared tunnel --url "http://localhost:${PORT}" \
  --no-autoupdate \
  --loglevel info 2>&1 | tee "$TUNNEL_LOG" &
TUNNEL_PID=$!

# Extract public URL from combined output log
PUBLIC_URL=""
echo -ne "  Waiting for tunnel"
for i in $(seq 1 40); do
  sleep 1
  PUBLIC_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1 || true)
  [[ -n "$PUBLIC_URL" ]] && break
  echo -ne "."
done
echo ""

if [[ -z "$PUBLIC_URL" ]]; then
  echo -e "  ${RED}❌ Could not get tunnel URL. cloudflared log:${END}"
  echo "──────────────────────────────────────────"
  tail -30 "$TUNNEL_LOG" 2>/dev/null || echo "(log empty)"
  echo "──────────────────────────────────────────"
  exit 1
fi

WEBHOOK_URL="${PUBLIC_URL}/webhook"

# ── Success! ──────────────────────────────────────────────────────────────────
banner "${GREEN}🚀 Everything is Running!${END}"
echo -e "  ${BOLD}Local server:${END}      http://localhost:${PORT}"
echo -e "  ${BOLD}Server log:${END}        /tmp/gohappy_server.log"
echo -e "  ${BOLD}Tunnel log:${END}        ${TUNNEL_LOG}"
echo -e "  ${BOLD}Public URL:${END}        ${CYAN}${PUBLIC_URL}${END}"
echo ""
echo -e "  ${BOLD}${GREEN}📋 Paste into Meta Developer Console → Webhook Callback URL:${END}"
echo -e "  ${BOLD}${YELLOW}     ${WEBHOOK_URL}${END}"
echo ""
echo -e "  ${BOLD}Verify Token:${END}      ${WHATSAPP_VERIFY_TOKEN}"
echo ""
echo -e "  ${CYAN}Run the test script in a SECOND terminal:${END}"
echo -e "    python test_send_receive.py"
echo -e "    python test_send_receive.py --send-to 919818646823"
echo ""
echo -e "  ${BOLD}Server logs (live):${END}"
echo "──────────────────────────────────────────"

# Live tail server log
tail -f /tmp/gohappy_server.log
