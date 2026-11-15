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

Use the gitignored **project-root .env** for Matterport, optional Claude, and ElevenLabs settings.
It contains ELEVENLABS_API_KEY, ELEVENLABS_AGENT_ID, ELEVENLABS_VOICE_SECRET_ID,
VOICE_TOOL_TOKEN, and VOICE_PUBLIC_URL. The backend, gateway, and setup helper read .env;
.env.local is not used. Existing process environment variables take priority.
Do not commit .env or paste its contents into chat.

Provisioning saves generated voice identifiers and credentials into .env.
Follow the main README's backend installation steps before running real room-location calls.

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

Use an eligible Photon SMS number or dedicated messaging line. Shared iMessage assignments
cannot receive voice. Use your personal phone to call the Photon number.

For an incoming-only demo:
1. In Photon Voice, create/select the number's SIP profile. Set media encryption to Auto.
2. In ElevenLabs Deploy > Phone Numbers > Import number > From SIP Trunk, enter the Photon number.
3. Set inbound media encryption to Allowed. Outbound configuration can stay disabled;
   an outbound SIP password is not required for incoming-only calls.
4. Assign HackPrinceton Photon Room Locator and the intended published branch to the imported number.
5. Set Photon inbound destination to sips:sip.rtc.elevenlabs.io:5061 with TLS.
   If you enable inbound SIP authentication, use the same inbound username/password on both platforms.
6. Call the Photon number, give the address, and describe the room. Watch the dashboard for
   backend-selected questions, room/floor estimates, and later corrections.
   The dashboard's Start call button remains the text/simulated entry point.

DEV_TEST_PHONE only prefills the dashboard caller field. After changing it in .env,
restart the FastAPI backend and refresh the browser; leave the gateway and tunnel running.
Restart both backend and gateway after changing VOICE_TOOL_TOKEN.

The agent and its saved tool bindings have been verified, with eight Python checks and one
gateway check passing. Live phone delivery and model inference were not verified by those
checks. See AGENTS.md at the repo root for the full architecture, environment reference, and demo steps.

## Checks without live calls

```powershell
cd C:\Users\Joshua\Projects\hackprinceton-mvp\backend
py -B -m unittest discover -s tests -p test_voice_bridge.py
cd ..\voice
npm test
```

Bridge tests use a fake session manager; pipeline tests use deterministic model outputs. They do not exercise live Matterport or model inference.
The agent speaks demo wording and does not claim to have contacted emergency services.

Sources:
- https://photon.codes/docs/beta/cookbooks/voice/elevenlabs
- https://elevenlabs.io/docs/eleven-agents/customization/tools/webhook-tools
- https://elevenlabs.io/docs/eleven-agents/customization/personalization/dynamic-variables
- https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/

Configure the ElevenLabs post-call transcription webhook at `/api/voice/post-call` on the voice gateway public URL. Set `ELEVENLABS_POST_CALL_SECRET` in the backend environment to the webhook signing secret, then restart the backend. Audio delivery is optional and should be left off for the demo.
