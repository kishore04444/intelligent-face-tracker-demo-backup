# VisitorIQ Dashboard

Professional local dashboard for the Intelligent Face Tracker.

## Live data
The UI reads the Python runtime APIs directly:
- `/api/stats` — current-session unique visitors, occupancy, entries and exits
- `/api/events` — current-session entry/exit events
- `/api/visitors` — persistent visitor registry
- `/api/stream` — live annotated camera feed
- `/api/status` — active source and processing state
- `POST /api/upload` — upload and process a video (512 MB maximum)
- `POST /api/source` — select a camera index or network stream URL
- `POST /api/browser-frame` — submit JPEG frames from a browser-authorized device camera
- `POST /api/browser-stop` — stop browser-camera ingestion
- `/api/visitors/<id>/image` — latest saved event face image for a visitor
- `/api/unrecognized-people` — permanent sightings history, independent of
  visitor and entry/exit data
- `/api/unrecognized-people/<id>/image` — saved image for a sighting
- `POST /api/reset` — queue a confirmed client-side reset of visitor identities,
  embeddings, events, and event-linked face images

The event API returns the latest 200 session events; session totals are
aggregated directly in SQLite so dashboard refresh work does not grow with
the event history. Identity templates are normalized and cached in memory for
fast matching, and that cache is cleared together with SQLite by the reset.
For browser camera access on another device, deploy behind HTTPS; browsers do
not grant camera permissions to plain HTTP LAN origins.

The server binds to `127.0.0.1` by default. To accept another device, set
`DASHBOARD_HOST=0.0.0.0`; for camera permissions, also configure trusted HTTPS
with `DASHBOARD_TLS_CERT` and `DASHBOARD_TLS_KEY` (or use an HTTPS reverse
proxy). The dashboard has no authentication, so do not expose it to an
untrusted network.

Visitor cards load their latest saved event face image. If the visitor has no
saved image, the card uses a generated visitor-ID placeholder.

The tracker opens RTSP sources over TCP first and falls back to UDP. If the
stream cannot be opened, the previously active source remains selected and the
status message advises checking reachability, stream path, and credentials.

## Design goals
- No hard-coded visitor metrics on the overview
- Current-session metrics separated from persistent visitor history
- Live camera feed from the Python tracker
- Back-facing / face-not-visible person boxes (detection only; no identity match)
- Permanent unrecognized sightings history, separate from visitor and event
  records; sightings do not count as entries/exits and reset with Reset database
- Derived operational metrics such as return visits, peak occupancy, average completed visit duration and exit rate
- Responsive dark operations-console style
