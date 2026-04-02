"""競輪予想 Web アプリケーション (FastAPI + Jinja2 + htmx)"""
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.common.database import (
    Odds, Player, Race, RaceEntry, Racecourse, get_session, init_db, seed_racecourses,
)
from src.predictor.model import KeirinPredictor

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="競輪予想システム")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.on_event("startup")
def startup():
    init_db()
    seed_racecourses()


def _render(request: Request, template: str, context: dict) -> HTMLResponse:
    """テンプレートをレンダリング (Starlette新旧API対応)"""
    return templates.TemplateResponse(request, template, context)


# ------------------------------------------------------------------
# ダッシュボード
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    session = get_session()
    try:
        races_total = session.query(Race).count()
        races_finished = session.query(Race).filter(Race.status == "finished").count()
        entries_total = session.query(RaceEntry).count()
        entries_with_result = session.query(RaceEntry).filter(
            RaceEntry.finish_position.isnot(None)
        ).count()
        players_total = session.query(Player).count()
        odds_total = session.query(Odds).count()

        # 本日のレース
        today = date.today()
        today_races = (
            session.query(Race)
            .filter(Race.race_date == today)
            .order_by(Race.racecourse_id, Race.race_number)
            .all()
        )

        today_race_list = []
        for race in today_races:
            rc = session.query(Racecourse).filter_by(id=race.racecourse_id).first()
            n_entries = session.query(RaceEntry).filter_by(race_id=race.id).count()
            today_race_list.append({
                "id": race.id,
                "venue": rc.name if rc else "不明",
                "race_number": race.race_number,
                "race_name": race.race_name or "",
                "grade": race.grade or "",
                "status": race.status,
                "n_entries": n_entries,
            })

        # 直近の開催日一覧
        recent_dates = (
            session.query(Race.race_date)
            .distinct()
            .order_by(Race.race_date.desc())
            .limit(10)
            .all()
        )
        recent_dates = [r[0] for r in recent_dates]

        return _render(request, "dashboard.html", {
            "stats": {
                "races_total": races_total,
                "races_finished": races_finished,
                "entries_total": entries_total,
                "entries_with_result": entries_with_result,
                "players_total": players_total,
                "odds_total": odds_total,
            },
            "today": today.isoformat(),
            "today_races": today_race_list,
            "recent_dates": recent_dates,
        })
    finally:
        session.close()


# ------------------------------------------------------------------
# レース一覧 (日付指定)
# ------------------------------------------------------------------

@app.get("/races", response_class=HTMLResponse)
def races_by_date(request: Request, d: str = Query(default="")):
    target = d or date.today().isoformat()
    session = get_session()
    try:
        dt = datetime.strptime(target, "%Y-%m-%d").date()
        races = (
            session.query(Race)
            .filter(Race.race_date == dt)
            .order_by(Race.racecourse_id, Race.race_number)
            .all()
        )

        race_list = []
        for race in races:
            rc = session.query(Racecourse).filter_by(id=race.racecourse_id).first()
            n_entries = session.query(RaceEntry).filter_by(race_id=race.id).count()
            race_list.append({
                "id": race.id,
                "venue": rc.name if rc else "不明",
                "race_number": race.race_number,
                "race_name": race.race_name or "",
                "grade": race.grade or "",
                "status": race.status,
                "n_entries": n_entries,
            })

        # htmx partial
        if request.headers.get("HX-Request"):
            return _render(request, "_race_list.html", {
                "races": race_list,
                "target_date": target,
            })

        return _render(request, "races.html", {
            "races": race_list,
            "target_date": target,
        })
    finally:
        session.close()


# ------------------------------------------------------------------
# レース詳細 + 予測
# ------------------------------------------------------------------

@app.get("/race/{race_id}", response_class=HTMLResponse)
def race_detail(request: Request, race_id: int, mode: str = Query(default="accuracy")):
    session = get_session()
    try:
        race = session.query(Race).filter_by(id=race_id).first()
        if not race:
            return HTMLResponse("<h2>レースが見つかりません</h2>", status_code=404)

        rc = session.query(Racecourse).filter_by(id=race.racecourse_id).first()
        entries = (
            session.query(RaceEntry)
            .filter_by(race_id=race.id)
            .order_by(RaceEntry.car_number)
            .all()
        )

        entry_list = []
        for e in entries:
            player = session.query(Player).filter_by(id=e.player_id).first() if e.player_id else None
            entry_list.append({
                "car_number": e.car_number,
                "frame_number": e.frame_number,
                "player_name": player.name if player else "不明",
                "grade": player.grade if player else "",
                "prefecture": player.prefecture if player else "",
                "win_rate": e.win_rate or 0,
                "second_rate": e.second_rate or 0,
                "third_rate": e.third_rate or 0,
                "avg_start": e.avg_start,
                "line_group": e.line_group or "",
                "line_label": e.line_label or "",
                "comment": e.comment or "",
                "finish_position": e.finish_position,
                "finish_time": e.finish_time or "",
                "win_technique": e.win_technique or "",
            })

        # 予測
        predictor = KeirinPredictor(mode=mode)
        predictions = predictor.predict(session, race)
        bets = predictor.suggest_bets(predictions, budget=1000)

        return _render(request, "race_detail.html", {
            "race": {
                "id": race.id,
                "date": race.race_date.isoformat(),
                "venue": rc.name if rc else "不明",
                "race_number": race.race_number,
                "race_name": race.race_name or "",
                "grade": race.grade or "",
                "status": race.status,
                "distance": race.distance,
                "is_girl": race.is_girl,
            },
            "entries": entry_list,
            "predictions": predictions,
            "bets": bets,
            "mode": mode,
        })
    finally:
        session.close()


# ------------------------------------------------------------------
# 予測 API (htmx partial)
# ------------------------------------------------------------------

@app.get("/predict/{race_id}", response_class=HTMLResponse)
def predict_partial(request: Request, race_id: int, mode: str = Query(default="accuracy")):
    session = get_session()
    try:
        race = session.query(Race).filter_by(id=race_id).first()
        if not race:
            return HTMLResponse("<p>レースが見つかりません</p>")

        predictor = KeirinPredictor(mode=mode)
        predictions = predictor.predict(session, race)
        bets = predictor.suggest_bets(predictions, budget=1000)

        return _render(request, "_predictions.html", {
            "predictions": predictions,
            "bets": bets,
            "mode": mode,
            "race_id": race_id,
        })
    finally:
        session.close()
