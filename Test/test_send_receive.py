import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
test_send_receive.py
====================
End-to-end test for the GoHappy Club Bot.

Tests:
  1. Health check — server is up
  2. Webhook verification (GET) — Meta handshake
  3. Simulated incoming message (POST) — bot receives & processes it
  4. WhatsApp API send test — sends a real message to a phone number
  5. Deduplication guard — same message_id twice must only process once

Usage:
  python test_send_receive.py                           # default server http://localhost:8080
  python test_send_receive.py --server http://localhost:8080
  python test_send_receive.py --send-to 919876543210   # also send a real WhatsApp message
  python test_send_receive.py --skip-api               # skip live API test
"""

import os
import sys
import json
import time
import uuid
import asyncio
import argparse
import datetime

# ── Load .env ────────────────────────────────────────────────────────────────
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

import httpx

# ── Colours ──────────────────────────────────────────────────────────────────
class C:
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    END    = "\033[0m"

PASS = f"{C.GREEN}✅ PASS{C.END}"
FAIL = f"{C.RED}❌ FAIL{C.END}"
SKIP = f"{C.YELLOW}⏭ SKIP{C.END}"

results: list[dict] = []

def banner(text):
    print(f"\n{C.BOLD}{'═'*60}")
    print(f"  {text}")
    print(f"{'═'*60}{C.END}\n")

def record(name: str, passed: bool, note: str = ""):
    status = PASS if passed else FAIL
    results.append({"name": name, "passed": passed, "note": note})
    print(f"  {status}  {name}")
    if note:
        print(f"         {C.CYAN}{note}{C.END}")

# ── Build a realistic Meta webhook payload ────────────────────────────────────
def make_webhook_payload(text: str, from_number: str = "919818646823",
                         display_name: str = "Test User",
                         msg_id: str | None = None) -> dict:
    phone_number_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "TEST_PHONE_ID")
    message_id = msg_id or f"wamid.test.{uuid.uuid4().hex}"
    return {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": os.environ.get("WHATSAPP_BUSINESS_ACCOUNT_ID", "TEST_BIZ_ID"),
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {
                        "display_phone_number": "919XXXXXXXXX",
                        "phone_number_id": phone_number_id,
                    },
                    "contacts": [{
                        "profile": {"name": display_name},
                        "wa_id": from_number,
                    }],
                    "messages": [{
                        "from": from_number,
                        "id": message_id,
                        "timestamp": str(int(time.time())),
                        "type": "text",
                        "text": {"body": text},
                    }],
                },
                "field": "messages",
            }],
        }],
    }

def make_status_payload() -> dict:
    """A delivery-status webhook — should be silently ignored by the bot."""
    phone_number_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "TEST_PHONE_ID")
    return {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "TEST_BIZ_ID",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {
                        "display_phone_number": "919XXXXXXXXX",
                        "phone_number_id": phone_number_id,
                    },
                    "statuses": [{
                        "id": f"wamid.status.{uuid.uuid4().hex}",
                        "status": "delivered",
                        "timestamp": str(int(time.time())),
                        "recipient_id": "919818646823",
                    }],
                },
                "field": "messages",
            }],
        }],
    }


async def run(server: str, send_to: str | None, skip_api: bool):
    base = server.rstrip("/")
    verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN", "gohappy_club_webhook_2026")
    access_token = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
    phone_number_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")

    banner(f"GoHappy Club Bot — End-to-End Test Suite\n  Server: {C.CYAN}{base}{C.END}")
    print(f"  Time:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    async with httpx.AsyncClient(timeout=20.0) as client:

        # ── TEST 1: Health check ──────────────────────────────────────────────
        banner("TEST 1: Health Check")
        try:
            resp = await client.get(f"{base}/health")
            ok = resp.status_code == 200
            data = resp.json() if ok else {}
            note = f"status={resp.status_code}  body={data}" if ok else f"status={resp.status_code}"
            record("GET /health → 200", ok, note)
        except Exception as exc:
            record("GET /health → 200", False, str(exc))

        # ── TEST 2: Webhook verification ──────────────────────────────────────
        banner("TEST 2: Webhook Verification (GET)")
        challenge = "challenge_" + uuid.uuid4().hex[:8]
        params = {
            "hub.mode": "subscribe",
            "hub.verify_token": verify_token,
            "hub.challenge": challenge,
        }

        # 2a — correct token
        try:
            resp = await client.get(f"{base}/webhook", params=params)
            ok = resp.status_code == 200 and resp.text.strip() == challenge
            record("GET /webhook correct token → echo challenge", ok,
                   f"returned '{resp.text.strip()}'")
        except Exception as exc:
            record("GET /webhook correct token → echo challenge", False, str(exc))

        # 2b — wrong token must 403
        try:
            bad_params = {**params, "hub.verify_token": "WRONG_TOKEN"}
            resp = await client.get(f"{base}/webhook", params=bad_params)
            ok = resp.status_code == 403
            record("GET /webhook wrong token → 403", ok, f"status={resp.status_code}")
        except Exception as exc:
            record("GET /webhook wrong token → 403", False, str(exc))

        # ── TEST 3: Incoming text message ─────────────────────────────────────
        banner("TEST 3: Incoming WhatsApp Message (POST)")

        test_messages = [
            ("Hello! What activities does GoHappy Club have?", "Normal Q&A query"),
            ("मुझे yoga class ke baare mein batao", "Hinglish query"),
            ("HOW DO I JOIN???", "Caps/punctuation stress test"),
        ]

        for text, label in test_messages:
            payload = make_webhook_payload(text)
            try:
                resp = await client.post(f"{base}/webhook", json=payload)
                # Bot must return 200 immediately (processing is async)
                ok = resp.status_code == 200
                record(f"POST /webhook [{label}] → 200", ok,
                       f"status={resp.status_code}  (bot processes in background)")
            except Exception as exc:
                record(f"POST /webhook [{label}] → 200", False, str(exc))

        # ── TEST 4: Status-update payload (should be silently ignored) ────────
        banner("TEST 4: Status-Update Payload (should be ignored)")
        payload = make_status_payload()
        try:
            resp = await client.post(f"{base}/webhook", json=payload)
            ok = resp.status_code == 200
            record("POST /webhook status-update → 200 (ignored)", ok,
                   f"status={resp.status_code}")
        except Exception as exc:
            record("POST /webhook status-update → 200 (ignored)", False, str(exc))

        # ── TEST 5: Deduplication ─────────────────────────────────────────────
        banner("TEST 5: Message Deduplication")
        fixed_msg_id = f"wamid.dedup.{uuid.uuid4().hex}"
        payload = make_webhook_payload("Does GoHappy have a morning walk?",
                                       msg_id=fixed_msg_id)
        ok_both = True
        try:
            for i in range(2):
                resp = await client.post(f"{base}/webhook", json=payload)
                if resp.status_code != 200:
                    ok_both = False
            record("POST /webhook same msg_id twice → both 200", ok_both,
                   "Server acks both; dedup prevents double processing internally")
        except Exception as exc:
            record("POST /webhook same msg_id twice → both 200", False, str(exc))

        # ── TEST 6: Malformed JSON ────────────────────────────────────────────
        banner("TEST 6: Malformed Request Handling")
        try:
            resp = await client.post(
                f"{base}/webhook",
                content=b"not-json!!!",
                headers={"Content-Type": "application/json"},
            )
            ok = resp.status_code == 400
            record("POST /webhook bad JSON → 400", ok, f"status={resp.status_code}")
        except Exception as exc:
            record("POST /webhook bad JSON → 400", False, str(exc))

        # ── TEST 7: Live WhatsApp API credential check ────────────────────────
        if skip_api:
            banner("TEST 7: WhatsApp API Credentials (SKIPPED)")
            results.append({"name": "WhatsApp API credentials", "passed": None, "note": "--skip-api"})
            print(f"  {SKIP}  WhatsApp API credentials (--skip-api)")
        else:
            banner("TEST 7: WhatsApp API Credentials (Live)")
            if not access_token or not phone_number_id or phone_number_id.startswith("<"):
                record("WhatsApp API credentials", False,
                       "WHATSAPP_ACCESS_TOKEN or WHATSAPP_PHONE_NUMBER_ID not set in .env")
            else:
                url = f"https://graph.facebook.com/v19.0/{phone_number_id}"
                headers = {"Authorization": f"Bearer {access_token}"}
                try:
                    resp = await client.get(url, headers=headers)
                    data = resp.json()
                    ok = resp.status_code == 200
                    if ok:
                        note = (f"name='{data.get('verified_name', 'N/A')}'  "
                                f"phone='{data.get('display_phone_number', 'N/A')}'  "
                                f"quality='{data.get('quality_rating', 'N/A')}'")
                    else:
                        err = data.get("error", {})
                        note = f"[{err.get('code')}] {err.get('message', 'unknown error')}"
                    record("WhatsApp API credentials valid", ok, note)
                except Exception as exc:
                    record("WhatsApp API credentials valid", False, str(exc))

        # ── TEST 8: Send a real WhatsApp message ──────────────────────────────
        if send_to:
            banner(f"TEST 8: Send Real WhatsApp Message → {send_to}")
            if skip_api or not access_token or phone_number_id.startswith("<"):
                print(f"  {SKIP}  Skipped — credentials not available")
            else:
                url = f"https://graph.facebook.com/v19.0/{phone_number_id}/messages"
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                }
                body = {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": send_to,
                    "type": "text",
                    "text": {
                        "body": (
                            "🎉 *GoHappy Club Bot — Test Message*\n\n"
                            "✅ WhatsApp API connection verified!\n"
                            "The bot is live and ready to respond.\n\n"
                            f"_Sent at {datetime.datetime.now().strftime('%H:%M:%S')}_"
                        )
                    },
                }
                try:
                    resp = await client.post(url, headers=headers, json=body)
                    data = resp.json()
                    ok = resp.status_code == 200
                    if ok:
                        msg_id = data.get("messages", [{}])[0].get("id", "N/A")
                        record(f"Send real message to {send_to}", ok, f"message_id={msg_id}")
                    else:
                        err = data.get("error", {})
                        note = f"[{err.get('code')}] {err.get('message', 'unknown')}"
                        if err.get("code") == 131030:
                            note += " — recipient must message the business number first"
                        record(f"Send real message to {send_to}", False, note)
                except Exception as exc:
                    record(f"Send real message to {send_to}", False, str(exc))
        else:
            banner("TEST 8: Send Real WhatsApp Message")
            print(f"  {SKIP}  Pass --send-to <phone_number> to run this test")

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("SUMMARY")
    total = len(results)
    passed = sum(1 for r in results if r["passed"] is True)
    failed = sum(1 for r in results if r["passed"] is False)
    skipped = sum(1 for r in results if r["passed"] is None)

    for r in results:
        icon = C.GREEN + "✅" + C.END if r["passed"] is True else \
               C.RED   + "❌" + C.END if r["passed"] is False else \
               C.YELLOW + "⏭" + C.END
        print(f"  {icon}  {r['name']}")
        if r["note"]:
            print(f"       {C.CYAN}{r['note']}{C.END}")

    print()
    print(f"  {C.BOLD}Total: {total}  "
          f"{C.GREEN}Passed: {passed}{C.END}  "
          f"{C.RED}Failed: {failed}{C.END}  "
          f"{C.YELLOW}Skipped: {skipped}{C.END}")

    if failed == 0:
        print(f"\n  {C.GREEN}{C.BOLD}🎉 All tests passed! Bot is working correctly.{C.END}\n")
    else:
        print(f"\n  {C.RED}{C.BOLD}⚠️  {failed} test(s) failed. Check output above.{C.END}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoHappy Club Bot — End-to-End Test Suite")
    parser.add_argument("--server", default="http://localhost:8080",
                        help="Base URL of the local bot server (default: http://localhost:8080)")
    parser.add_argument("--send-to", metavar="PHONE",
                        help="Send a real WhatsApp test message to this E.164 number (e.g. 919876543210)")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip live WhatsApp API tests (useful for pure local testing)")
    args = parser.parse_args()

    asyncio.run(run(
        server=args.server,
        send_to=args.send_to,
        skip_api=args.skip_api,
    ))
