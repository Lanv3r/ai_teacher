from flask import Flask, render_template, request, redirect, url_for, session
from datetime import date
import uuid

from .scheduler import schedule_tasks


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "dev-secret-change-me"

    @app.route("/")
    def index():
        data = session.get("data") or {}
        tasks = data.get("tasks", [])
        prefs = data.get("preferences", default_preferences())
        return render_template("index.html", tasks=tasks, prefs=prefs)

    @app.route("/add-task", methods=["POST"])
    def add_task():
        data = session.get("data") or {"tasks": [], "preferences": default_preferences()}
        tasks = data["tasks"]

        name = request.form.get("name", "").strip()
        subject = request.form.get("subject", "General").strip() or "General"
        minutes = int(request.form.get("minutes", 0) or 0)
        deadline = request.form.get("deadline") or date.today().isoformat()

        if name and minutes > 0:
            tasks.append({
                "id": str(uuid.uuid4()),
                "name": name,
                "subject": subject,
                "minutes": minutes,
                "deadline": deadline,
            })
        
        data["tasks"] = tasks
        session["data"] = data
        return redirect(url_for("index"))

    @app.route("/delete-task/<task_id>")
    def delete_task(task_id):
        data = session.get("data") or {"tasks": [], "preferences": default_preferences()}
        data["tasks"] = [t for t in data["tasks"] if t["id"] != task_id]
        session["data"] = data
        return redirect(url_for("index"))

    @app.route("/save-preferences", methods=["POST"])
    def save_preferences():
        data = session.get("data") or {"tasks": [], "preferences": default_preferences()}
        prefs = data["preferences"]

        prefs["work_block_minutes"] = int(request.form.get("work_block_minutes", 50) or 50)
        prefs["short_break_minutes"] = int(request.form.get("short_break_minutes", 10) or 10)
        prefs["long_break_every_blocks"] = int(request.form.get("long_break_every_blocks", 4) or 4)
        prefs["long_break_minutes"] = int(request.form.get("long_break_minutes", 20) or 20)

        # Parse study windows for 7 days
        windows = []
        for wd in range(7):
            start = request.form.get(f"start_{wd}")
            end = request.form.get(f"end_{wd}")
            if start and end:
                windows.append({"weekday": wd, "start": start, "end": end})
        prefs["study_windows"] = windows

        data["preferences"] = prefs
        session["data"] = data
        return redirect(url_for("index"))

    @app.route("/schedule")
    def schedule_view():
        data = session.get("data") or {"tasks": [], "preferences": default_preferences()}
        tasks = data.get("tasks", [])
        prefs = data.get("preferences", default_preferences())
        sessions = schedule_tasks(tasks, prefs)
        return render_template("schedule.html", sessions=sessions, tasks={t["id"]: t for t in tasks})

    return app


def default_preferences():
    return {
        "study_windows": [
            {"weekday": 0, "start": "18:00", "end": "22:00"},
            {"weekday": 1, "start": "18:00", "end": "22:00"},
            {"weekday": 2, "start": "18:00", "end": "22:00"},
            {"weekday": 3, "start": "18:00", "end": "22:00"},
            {"weekday": 4, "start": "18:00", "end": "22:00"},
            {"weekday": 5, "start": "10:00", "end": "18:00"},
            {"weekday": 6, "start": "10:00", "end": "18:00"},
        ],
        "work_block_minutes": 50,
        "short_break_minutes": 10,
        "long_break_every_blocks": 4,
        "long_break_minutes": 20,
    }


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)


