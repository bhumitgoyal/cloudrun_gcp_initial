"""
test_whatsapp_api.py
====================
Verifies WhatsApp Cloud API credentials by:
  1. Fetching phone number details (GET /v19.0/{phone_number_id})
  2. Fetching business profile info
  3. Optionally sending a test message to a specified number

Uses the REAL credentials from .env — no mocks.
"""

import os
import sys
import asyncio
import json

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

GRAPH_API_BASE = "https://graph.facebook.com/v19.0"

# ── Colours ──────────────────────────────────────────────────────────────────
class C:
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    END    = "\033[0m"

def banner(text):
    print(f"\n{C.BOLD}{'═'*60}")
    print(f"  {text}")
    print(f"{'═'*60}{C.END}\n")


async def run():
    access_token     = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
    phone_number_id  = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
    business_acct_id = os.environ.get("WHATSAPP_BUSINESS_ACCOUNT_ID", "")

    banner("WhatsApp Cloud API — Credential Verification")

    print(f"  {C.BOLD}Phone Number ID:{C.END}         {phone_number_id}")
    print(f"  {C.BOLD}Business Account ID:{C.END}     {business_acct_id}")
    print(f"  {C.BOLD}Access Token:{C.END}            {access_token[:20]}...{access_token[-10:]}")
    print()

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:

        # ── TEST 1: Fetch Phone Number Details ───────────────────────────
        banner("TEST 1: Fetch Phone Number Details")
        url = f"{GRAPH_API_BASE}/{phone_number_id}"
        print(f"  GET {url}")
        try:
            resp = await client.get(url, headers=headers)
            print(f"  Status: {resp.status_code}")
            data = resp.json()
            print(f"  Response:\n{json.dumps(data, indent=4)}")

            if resp.status_code == 200:
                phone_display = data.get("display_phone_number", "N/A")
                verified_name = data.get("verified_name", "N/A")
                quality_rating = data.get("quality_rating", "N/A")
                status = data.get("code_verification_status", "N/A")
                print(f"\n  {C.GREEN}✅ CREDENTIALS VALID!{C.END}")
                print(f"  {C.CYAN}Phone Number:{C.END}    {phone_display}")
                print(f"  {C.CYAN}Verified Name:{C.END}   {verified_name}")
                print(f"  {C.CYAN}Quality Rating:{C.END}  {quality_rating}")
                print(f"  {C.CYAN}Status:{C.END}          {status}")
            else:
                print(f"\n  {C.RED}❌ FAILED — Check your access token and phone number ID.{C.END}")
                error = data.get("error", {})
                print(f"  Error: {error.get('message', 'Unknown')}")
                print(f"  Code:  {error.get('code', 'N/A')}")
                return
        except Exception as exc:
            print(f"  {C.RED}❌ Request failed: {exc}{C.END}")
            return

        # ── TEST 2: Fetch Business Profile ───────────────────────────────
        banner("TEST 2: Fetch WhatsApp Business Profile")
        url = f"{GRAPH_API_BASE}/{phone_number_id}/whatsapp_business_profile?fields=about,address,description,email,profile_picture_url,websites,vertical"
        print(f"  GET {url[:80]}...")
        try:
            resp = await client.get(url, headers=headers)
            print(f"  Status: {resp.status_code}")
            data = resp.json()
            print(f"  Response:\n{json.dumps(data, indent=4)}")

            if resp.status_code == 200:
                profile_data = data.get("data", [{}])[0] if data.get("data") else {}
                print(f"\n  {C.GREEN}✅ Business Profile Retrieved!{C.END}")
                print(f"  {C.CYAN}About:{C.END}        {profile_data.get('about', '(not set)')}")
                print(f"  {C.CYAN}Description:{C.END}  {profile_data.get('description', '(not set)')}")
                print(f"  {C.CYAN}Vertical:{C.END}    {profile_data.get('vertical', '(not set)')}")
            else:
                print(f"\n  {C.YELLOW}⚠️  Could not fetch business profile (may not be configured yet).{C.END}")
        except Exception as exc:
            print(f"  {C.YELLOW}⚠️  Business profile request failed: {exc}{C.END}")

        # ── TEST 3: Verify Message Template Access ───────────────────────
        banner("TEST 3: Check Message Template Access")
        url = f"{GRAPH_API_BASE}/{business_acct_id}/message_templates?limit=5"
        print(f"  GET {url}")
        try:
            resp = await client.get(url, headers=headers)
            print(f"  Status: {resp.status_code}")
            data = resp.json()

            if resp.status_code == 200:
                templates = data.get("data", [])
                print(f"\n  {C.GREEN}✅ Template access works! Found {len(templates)} template(s).{C.END}")
                for t in templates[:5]:
                    print(f"     - {t.get('name', 'N/A')} ({t.get('status', 'N/A')})")
            else:
                print(f"  Response:\n{json.dumps(data, indent=4)}")
                print(f"\n  {C.YELLOW}⚠️  Template access issue (this is optional).{C.END}")
        except Exception as exc:
            print(f"  {C.YELLOW}⚠️  Template request failed: {exc}{C.END}")

        # ── TEST 4: Send a test message (optional) ───────────────────────
        banner("TEST 4: Send Test Message")
        
        # Ask for recipient number
        test_number = input(f"  {C.BOLD}Enter a phone number to send a test message to (E.164 format, e.g. 919876543210),\n  or press Enter to skip: {C.END}").strip()
        
        if test_number:
            url = f"{GRAPH_API_BASE}/{phone_number_id}/messages"
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": test_number,
                "type": "text",
                "text": {"body": "🎉 Hello from GoHappy Club Bot! This is a test message to verify the WhatsApp API connection is working. If you can read this, the bot is ready! 🙏"}
            }
            print(f"\n  Sending to {test_number}...")
            try:
                resp = await client.post(url, headers=headers, json=payload)
                print(f"  Status: {resp.status_code}")
                data = resp.json()
                print(f"  Response:\n{json.dumps(data, indent=4)}")

                if resp.status_code == 200:
                    msg_id = data.get("messages", [{}])[0].get("id", "N/A")
                    print(f"\n  {C.GREEN}✅ MESSAGE SENT SUCCESSFULLY!{C.END}")
                    print(f"  {C.CYAN}Message ID:{C.END} {msg_id}")
                else:
                    print(f"\n  {C.RED}❌ Send failed.{C.END}")
                    error = data.get("error", {})
                    print(f"  Error: {error.get('message', 'Unknown')}")
                    code = error.get("code", 0)
                    if code == 131030:
                        print(f"\n  {C.YELLOW}💡 This likely means the recipient hasn't messaged this number first.")
                        print(f"     WhatsApp requires the user to send a message first before you can reply.")
                        print(f"     Ask them to send 'Hi' to the business number, then try again.{C.END}")
            except Exception as exc:
                print(f"  {C.RED}❌ Send failed: {exc}{C.END}")
        else:
            print(f"  {C.YELLOW}⏭️  Skipped (no number provided).{C.END}")

    # ── Summary ──────────────────────────────────────────────────────────
    banner("VERIFICATION COMPLETE")
    print(f"  {C.GREEN}Your WhatsApp API credentials are configured and verified.")
    print(f"  Next steps:")
    print(f"    1. Deploy to Cloud Run (see DEPLOY.md)")
    print(f"    2. Set your Cloud Run URL as the webhook callback in Meta Developer Console")
    print(f"    3. Subscribe to 'messages' webhook field")
    print(f"    4. Send a message to the business number to trigger the bot!{C.END}\n")


if __name__ == "__main__":
    asyncio.run(run())
