# Photon + ElevenLabs voice

The ElevenLabs agent is a voice interface to the existing FastAPI SessionManager. It does not
choose rooms or follow-up questions. The transcript goes to handle_reply, which runs
questions.parse_reply, the optional Claude reader or semantics.scores, followups.read_message,
CLIP evidence, Bayesian updates, floor checks, followups.choose, and in-room refinements.
The same WebSocket broadcasts update the dispatcher dashboard.

Voice sessions use channel="voice", so they do not send iMessages. They are bound to the
ElevenLabs conversation ID, not the phone number. Repeated tool requests reuse the recorded
answer result; repeated "yes" answers on different transcript turns are processed separately.
State is in memory, like existing sessions: run one backend worker and restart conversations
after a backend restart. Full voice turns are serialized per conversation.

## Local configuration

Onboarding creates the gitignored root .env.local with ELEVENLABS_API_KEY.
Provisioning also saves ELEVENLABS_AGENT_ID, ELEVENLABS_VOICE_SECRET_ID, VOICE_TOOL_TOKEN,
and VOICE_PUBLIC_URL there. Do not commit this file or paste its contents into chat.
The backend loads .env.local before .env; actual process environment variables take priority.

The existing Matterport and optional Claude credentials still belong in root .env.
This checkout currently needs its normal Python backend dependencies and Matterport credentials
before it can locate real rooms. Follow the main README's backend installation steps.

## Start the services

Keep the existing FastAPI backend (:8000) and dispatcher dashboard (:5173) running as usual.
Restart the backend after provisioning so it loads VOICE_TOOL_TOKEN and the new routes.

In a terminal, run the authenticated gateway:

```powershell
cd C:\Users\Joshua\Projects\hackprinceton-mvp\voice
npm install
npm start
```

The gateway binds only to 127.0.0.1:8788. It forwards POST /api/voice/start, /reply,
and /status to BACKEND_URL (default http://127.0.0.1:8000). Everything else is rejected.
Both the gateway and backend require the shared bearer credential.

In another terminal, create a temporary HTTPS tunnel:

```powershell
cloudflared tunnel --url http://127.0.0.1:8788 --no-autoupdate
```

If cloudflared is not on PATH, onboarding downloaded a temporary copy to:
C:\Users\Joshua\AppData\Local\Temp\elevenlabs-skills-hJr7VU\cloudflared.exe

Configure the agent using the HTTPS URL printed by cloudflared:

```powershell
cd C:\Users\Joshua\Projects\hackprinceton-mvp\voice
$env:NODE_OPTIONS='--use-system-ca'
npm run setup -- https://YOUR-HOST.trycloudflare.com
```

The setup command uses the ElevenLabs SDK to create the agent, or update the saved agent.
It stores the gateway bearer credential in ElevenLabs workspace secrets and configures three
server tools. It never starts audio or places a call. Re-run setup after every tunnel restart,
because Quick Tunnel hostnames change. A stopped tunnel cannot receive agent tool requests.
Quick Tunnels are intended for development; use a stable tunnel/deployment for a lasting URL.

## Connect the Photon number

Your current shared iMessage assignment cannot receive voice calls. First obtain an eligible
Photon SMS number, WhatsApp number, or dedicated iMessage line.

1. In Photon Voice, create/select a SIP profile for that number. Set media encryption to Auto.
2. Enable outbound using SHA-256. Save the generated SIP username and password securely.
3. In ElevenLabs Deploy > Phone Numbers, import the Photon number From SIP Trunk (E.164 format).
4. Set outbound Address to the server and TLS port displayed by Photon (e.g. sip.photon.codes:5061,
   without a sip: prefix), Transport TLS, media encryption Allowed, and Photon's SIP username/password.
5. Assign the agent named HackPrinceton Photon Room Locator to that imported number.
6. In Photon inbound settings, set sips:sip.rtc.elevenlabs.io:5061 with TLS.
   For the cookbook's initial setup, ElevenLabs inbound media is Allowed, allowed sources are
   0.0.0.0/0, and inbound SIP credentials are empty on both sides. If you enable inbound
   authentication, use the same separate inbound credentials on both platforms.
7. Call the Photon number from another phone. Give the building address and city, then describe
   the room. Check that the dashboard displays the session, backend-selected follow-up questions,
   room/floor estimates, and later corrections.

Photon SIP credentials are different from PHOTON_API_KEY and ELEVENLABS_API_KEY.
No Photon SIP settings or phone-number imports are performed by the agent setup command.

## Checks without live calls

```powershell
cd C:\Users\Joshua\Projects\hackprinceton-mvp\backend
py -B -m unittest discover -s tests -p test_voice_bridge.py
cd ..\voice
npm test
```

Transport tests use a fake session manager; they do not exercise live Matterport or model inference.
The agent speaks demo wording and does not claim to have contacted emergency services.

Sources:
- https://photon.codes/docs/beta/cookbooks/voice/elevenlabs
- https://elevenlabs.io/docs/eleven-agents/customization/tools/webhook-tools
- https://elevenlabs.io/docs/eleven-agents/customization/personalization/dynamic-variables
- https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
