import os
import time
from dotenv import load_dotenv
import json
from portkey_ai import Portkey

load_dotenv()
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")

# ── Replace with your saved config slug from app.portkey.ai ───────
PORTKEY_CONFIG_SLUG = os.getenv("PORTKEY_CONFIG_SLUG", "pc-retry-118369")  # e.g. "pc-retry-xxxxx"

GROQ_SLUG = "gkey"
GROQ_MODEL = f"@{GROQ_SLUG}/openai/gpt-oss-20b"

def section(title):
    print(f"\n{'='*62}")
    print(f" {title}")
    print(f"{'='*62}")

def timed_call(client, model, prompt, label=""):
    """Make a call and print timing + result."""
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        ms = (time.time() - t0) * 1000
        answer = r.choices[0].message.content[:250]
        print(f"  ✅ {label} — {ms:.0f}ms")
        print(f"     {answer}")
    except Exception as e:
        ms = (time.time() - t0) * 1000
        err = str(e)[:200]
        print(f"  ❌ {label} — {ms:.0f}ms")
        print(f"     Error: {err}")

# ══════════════════════════════════════════════════════════════════
# 1. WITHOUT CONFIG — plain call (baseline)
# ══════════════════════════════════════════════════════════════════
section("1. BASELINE — No retry/timeout config")
plain = Portkey(api_key=PORTKEY_API_KEY)
timed_call(plain, GROQ_MODEL, "Say hello in one sentence.", label="Plain call")

# ══════════════════════════════════════════════════════════════════
# 2. WITH SAVED CONFIG — retry + timeout enabled
# ══════════════════════════════════════════════════════════════════
if PORTKEY_CONFIG_SLUG:
    section("2. WITH SAVED CONFIG — retry + timeout")
    configured = Portkey(api_key=PORTKEY_API_KEY, config=PORTKEY_CONFIG_SLUG)

    # Normal request — should succeed within timeout
    timed_call(configured, GROQ_MODEL,
               "What is 2+2? Answer in one word.",
               label="Fast question (should succeed)")

    # Longer request — might be close to timeout
    timed_call(configured, GROQ_MODEL,
               "Write a detailed 500-word essay on quantum computing.",
               label="Long generation (may hit timeout)")

    # ══════════════════════════════════════════════════════════════
    # 3. INVALID MODEL — triggers retry on error
    # ══════════════════════════════════════════════════════════════
    section("3. RETRY BEHAVIOR — invalid model (expect retries then fail)")
    timed_call(configured, f"@{GROQ_SLUG}/openai/nonexistent-model-xyz",
               "Hello",
               label="Bad model (retries visible in Portkey logs)")

    # ══════════════════════════════════════════════════════════════
    # 4. METADATA — tag requests for dashboard filtering
    # ══════════════════════════════════════════════════════════════
    section("4. METADATA — tagged request for dashboard filtering")
    tagged = configured.with_options(
        metadata={
            "_user": "kartik",
            "experiment": "retry-timeout-test",
            "environment": "dev"
        }
    )
    timed_call(tagged, GROQ_MODEL,
               "What is Portkey AI gateway?",
               label="Tagged request")

    print(f"\n{'─'*62}")
    print(" Check app.portkey.ai → Logs to see:")
    print("   • Request latencies and retry attempts")
    print("   • Timeout behavior on long generations")
    print("   • Failed requests with retry count")
    print("   • Filter by metadata: experiment=retry-timeout-test")
    print(f"{'─'*62}")
else:
    section("⚠️  SKIPPED — No saved config slug found")
    print("  Set PORTKEY_CONFIG_SLUG in your .env file:")
    print('  PORTKEY_CONFIG_SLUG=pc-your-slug-here')
    print()
    print("  Steps:")
    print("  1. Go to app.portkey.ai → Configs → Create")
    print("  2. Paste this config:")
    config = {
        "strategy": {"mode": "single"},
        "targets": [{
            "virtual_key": "gkey",
            "override_params": {"model": "openai/gpt-oss-20b"}
        }],
        "request_timeout": 10000,
        "retry": {
            "attempts": 3,
            "on_status_codes": [408, 429, 500, 502, 503, 504]
        }
    }
    print(json.dumps(config, indent=2))
    print('  3. Save and add the slug to .env: PORTKEY_CONFIG_SLUG=pc-xxxxx')