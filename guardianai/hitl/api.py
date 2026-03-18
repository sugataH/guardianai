"""
guardianai.hitl.api
====================
HITL REST API — exposes the HITLQueue for human operators.

Endpoints:
    GET  /hitl/pending              — list all pending entries (urgent first)
    GET  /hitl/entry/{entry_id}     — get single entry details
    GET  /hitl/agent/{agent_id}     — all entries for a specific agent
    POST /hitl/review/{entry_id}    — mark as reviewed
    POST /hitl/resolve/{agent_id}   — resolve all entries + restore agent
    POST /hitl/escalate/{entry_id}  — escalate to security team
    GET  /hitl/snapshot             — full queue state

Usage (standalone):
    from guardianai.hitl.api import create_hitl_app
    app = create_hitl_app(supervisor, hitl_queue)
    app.run(host="0.0.0.0", port=5001)

Or mount as Blueprint in a larger Flask application.

Dependencies:
    pip install flask
"""

import logging
logger = logging.getLogger(__name__)

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logger.warning("Flask not installed. HITL API unavailable. pip install flask")


def create_hitl_app(supervisor, hitl_queue):
    """
    Create a Flask app wrapping the HITL queue and Supervisor.

    Args:
        supervisor:  SupervisorAgent instance
        hitl_queue:  HITLQueue instance

    Returns:
        Flask app (or None if Flask unavailable)
    """
    if not FLASK_AVAILABLE:
        logger.error("Cannot create HITL app: Flask not installed.")
        return None

    app = Flask("guardianai_hitl")

    @app.route("/hitl/pending", methods=["GET"])
    def list_pending():
        return jsonify(hitl_queue.list_pending())

    @app.route("/hitl/entry/<entry_id>", methods=["GET"])
    def get_entry(entry_id):
        entry = hitl_queue.get_entry(entry_id)
        if not entry:
            return jsonify({"error": "not found"}), 404
        return jsonify(entry)

    @app.route("/hitl/agent/<agent_id>", methods=["GET"])
    def get_agent_entries(agent_id):
        return jsonify(hitl_queue.get_agent_entries(agent_id))

    @app.route("/hitl/review/<entry_id>", methods=["POST"])
    def review_entry(entry_id):
        body     = request.get_json() or {}
        reviewer = body.get("reviewer", "unknown_operator")
        notes    = body.get("notes")
        result   = hitl_queue.review(entry_id, reviewer, notes)
        if not result:
            return jsonify({"error": "entry not found"}), 404
        return jsonify(result)

    @app.route("/hitl/resolve/<agent_id>", methods=["POST"])
    def resolve_agent(agent_id):
        body         = request.get_json() or {}
        resolved_by  = body.get("resolved_by", "HITL_OPERATOR")
        status       = supervisor.restore_agent(agent_id, resolved_by)
        return jsonify(status)

    @app.route("/hitl/escalate/<entry_id>", methods=["POST"])
    def escalate_entry(entry_id):
        body         = request.get_json() or {}
        escalated_by = body.get("escalated_by", "unknown_operator")
        result       = hitl_queue.escalate(entry_id, escalated_by)
        if not result:
            return jsonify({"error": "entry not found"}), 404
        return jsonify(result)

    @app.route("/hitl/snapshot", methods=["GET"])
    def snapshot():
        return jsonify({
            "hitl":       hitl_queue.snapshot(),
            "supervisor": supervisor.snapshot(),
        })

    @app.route("/hitl/agent_status/<agent_id>", methods=["GET"])
    def agent_status(agent_id):
        return jsonify(supervisor.get_agent_status(agent_id))

    return app
