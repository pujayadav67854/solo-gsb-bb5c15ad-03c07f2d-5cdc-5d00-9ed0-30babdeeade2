"""Regression tests: ``points`` must be a strict signed 64-bit integer.

Covers ``POST /events``, ``POST /events/batch`` and ``POST /events/preview``:
booleans, floats, numeric strings and out-of-range values must all be 422
instead of being coerced or blowing up inside SQLite; the signed-64 boundary
values must still write, replay, correct and preview.
"""

import pytest
from fastapi.testclient import TestClient

import app.db as db_module
from app.main import app

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient backed by an isolated SQLite file."""
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "points-test.db"))
    db_module.init_db()
    return TestClient(app)


def _ev(event_id, points, participant_id="alice", correction_of=None):
    payload = {
        "event_id": event_id,
        "participant_id": participant_id,
        "points": points,
    }
    if correction_of is not None:
        payload["correction_of"] = correction_of
    return payload


# Values that lax ``int`` coercion used to accept, or that overflow SQLite.
INVALID_POINTS = [
    True,
    False,
    0.0,          # float exactly equal to an int
    1.5,          # non-integral float
    "10",         # numeric string
    "-3",         # numeric string, negative
    INT64_MAX + 1,
    INT64_MIN - 1,
    10**100,      # arbitrarily large bigint
]


def _assert_points_error(resp):
    assert resp.status_code == 422, resp.text
    locs = [".".join(str(p) for p in err["loc"] if p != "body") for err in resp.json()["detail"]]
    assert any(loc.endswith("points") for loc in locs), resp.json()


# --- POST /events ---------------------------------------------------------


@pytest.mark.parametrize("bad", INVALID_POINTS)
def test_post_event_rejects_non_strict_or_overflow_points(client, bad):
    # floats must travel as JSON numbers, not be re-coerced by the test client
    resp = client.post("/events", content=_json_bytes(_ev("bad", bad)),
                       headers={"Content-Type": "application/json"})
    _assert_points_error(resp)
    # rejection must not store anything or consume a revision
    assert client.get("/events").json() == []
    assert client.get("/leaderboard").json()["revision"] == 0


def test_post_event_rejects_bool_sent_as_json(client):
    # explicit raw body to be sure ``true`` reaches the app, not Python True
    resp = client.post(
        "/events",
        content=b'{"event_id":"b","participant_id":"alice","points":true}',
        headers={"Content-Type": "application/json"},
    )
    _assert_points_error(resp)


def test_post_event_accepts_zero_and_negative_points(client):
    assert client.post("/events", json=_ev("zero", 0)).status_code == 201
    assert client.post("/events", json=_ev("neg", -5)).status_code == 201
    assert client.get("/events/zero").json()["points"] == 0
    assert client.get("/events/neg").json()["points"] == -5


def test_post_event_boundary_points_write_read_and_replay(client):
    first_min = client.post("/events", json=_ev("min", INT64_MIN))
    assert first_min.status_code == 201, first_min.text
    assert first_min.json()["points"] == INT64_MIN

    first_max = client.post("/events", json=_ev("max", INT64_MAX, participant_id="bob"))
    assert first_max.status_code == 201, first_max.text
    assert first_max.json()["points"] == INT64_MAX

    # values survive a real round-trip through SQLite
    assert client.get("/events/min").json()["points"] == INT64_MIN
    assert client.get("/events/max").json()["points"] == INT64_MAX

    # idempotent replay of boundary payloads: 200 with the original result
    replay_min = client.post("/events", json=_ev("min", INT64_MIN))
    assert replay_min.status_code == 200
    assert replay_min.json() == first_min.json()
    replay_max = client.post("/events", json=_ev("max", INT64_MAX, participant_id="bob"))
    assert replay_max.status_code == 200
    assert replay_max.json() == first_max.json()
    assert client.get("/leaderboard").json()["revision"] == 2


def test_post_event_boundary_points_support_correction(client):
    assert client.post("/events", json=_ev("root", INT64_MAX)).status_code == 201
    corrected = client.post(
        "/events", json=_ev("fix", INT64_MIN, correction_of="root")
    )
    assert corrected.status_code == 201, corrected.text
    assert corrected.json()["points"] == INT64_MIN
    # only the correcting tip counts at the new revision
    lb = client.get("/leaderboard").json()
    alice = next(e for e in lb["entries"] if e["participant_id"] == "alice")
    assert alice["total_points"] == INT64_MIN


# --- POST /events/batch ---------------------------------------------------


@pytest.mark.parametrize("bad", INVALID_POINTS)
def test_batch_rejects_non_strict_or_overflow_points(client, bad):
    # one perfectly valid item next to the bad one: the whole request must
    # fail at validation time, before any transaction is opened
    resp = client.post(
        "/events/batch",
        content=_json_bytes([_ev("good", 1), _ev("bad", bad)]),
        headers={"Content-Type": "application/json"},
    )
    _assert_points_error(resp)
    assert client.get("/events").json() == []
    assert client.get("/events/good").status_code == 404
    assert client.get("/leaderboard").json()["revision"] == 0


def test_batch_invalid_item_consumes_no_revision_after_valid_writes(client):
    ok = client.post("/events/batch", json=[_ev("ok1", 10), _ev("ok2", 20)])
    assert ok.status_code == 201

    bad = client.post(
        "/events/batch",
        json=[_ev("new1", 1), _ev("new2", True), _ev("new3", INT64_MAX + 1)],
    )
    assert bad.status_code == 422
    assert client.get("/events/new1").status_code == 404
    assert client.get("/events/new2").status_code == 404
    assert client.get("/events/new3").status_code == 404
    assert client.get("/leaderboard").json()["revision"] == 2


def test_batch_boundary_points_write_and_replay(client):
    first = client.post(
        "/events/batch",
        json=[
            _ev("b-min", INT64_MIN),
            _ev("b-max", INT64_MAX, participant_id="bob"),
        ],
    )
    assert first.status_code == 201, first.text
    assert [e["points"] for e in first.json()] == [INT64_MIN, INT64_MAX]
    assert [e["revision"] for e in first.json()] == [1, 2]

    # full idempotent replay with the same boundary payloads
    replay = client.post(
        "/events/batch",
        json=[
            _ev("b-min", INT64_MIN),
            _ev("b-max", INT64_MAX, participant_id="bob"),
        ],
    )
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert client.get("/leaderboard").json()["revision"] == 2


# --- POST /events/preview -------------------------------------------------


@pytest.mark.parametrize("bad", INVALID_POINTS)
def test_preview_rejects_non_strict_or_overflow_points(client, bad):
    resp = client.post(
        "/events/preview",
        content=_json_bytes(
            {"base_revision": 0, "events": [_ev("good", 1), _ev("bad", bad)]}
        ),
        headers={"Content-Type": "application/json"},
    )
    _assert_points_error(resp)
    # previews never write; a validation failure must leave revision at 0
    assert client.get("/leaderboard").json()["revision"] == 0


def test_preview_boundary_points_project(client):
    resp = client.post(
        "/events/preview",
        json={
            "base_revision": 0,
            "events": [
                _ev("p-min", INT64_MIN, participant_id="alice"),
                _ev("p-max", INT64_MAX, participant_id="bob"),
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    projected = resp.json()["projected_leaderboard"]
    assert projected["revision"] == 2
    assert [(e["participant_id"], e["total_points"]) for e in projected["entries"]] == [
        ("bob", INT64_MAX),
        ("alice", INT64_MIN),
    ]
    # dry run only: nothing landed in the ledger
    assert client.get("/events").json() == []
    assert client.get("/leaderboard").json()["revision"] == 0


def test_preview_boundary_points_in_correction_draft(client):
    # baseline event at the upper boundary, then a draft correcting it to the
    # lower boundary: only the draft tip must be projected
    assert client.post("/events", json=_ev("base", INT64_MAX)).status_code == 201
    resp = client.post(
        "/events/preview",
        json={
            "base_revision": 1,
            "events": [_ev("draft", INT64_MIN, correction_of="base")],
        },
    )
    assert resp.status_code == 200, resp.text
    projected = resp.json()["projected_leaderboard"]
    alice = next(e for e in projected["entries"] if e["participant_id"] == "alice")
    assert alice["total_points"] == INT64_MIN
    assert [ev["event_id"] for ev in alice["effective_events"]] == ["draft"]
    assert client.get("/leaderboard").json()["revision"] == 1


# --- helpers --------------------------------------------------------------


def _json_bytes(payload):
    """Serialize without Python's bool/int distinctions being smoothed over.

    ``json.dumps`` renders ``True``/``False`` as ``true``/``false`` and keeps
    ints vs floats distinct, which is exactly the wire format we care about.
    """
    import json

    return json.dumps(payload).encode("utf-8")
