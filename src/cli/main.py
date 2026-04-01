"""競輪予想 CLI

使用例:
    python -m src.cli.main scrape --date 2026-04-01
    python -m src.cli.main predict --date 2026-04-01 --mode accuracy
    python -m src.cli.main train
    python -m src.cli.main status
"""
import sys
from datetime import date, datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.common.database import Race, RaceEntry, Racecourse, get_session, init_db, seed_racecourses
from src.parser.store import store_odds, store_race_entries, store_results
from src.predictor.model import KeirinPredictor
from src.scraper.kdreams import KdreamsScraper
from src.scraper.winticket import WinticketScraper

console = Console()


@click.group()
def cli():
    """競輪予想システム"""
    init_db()
    seed_racecourses()


@cli.command()
@click.option("--date", "target_date", default=None, help="対象日 (YYYY-MM-DD) デフォルト: 今日")
@click.option("--source", type=click.Choice(["winticket", "kdreams"]), default="kdreams", help="データソース")
@click.option("--with-odds", is_flag=True, help="オッズも取得する")
@click.option("--with-results", is_flag=True, help="結果も取得する")
def scrape(target_date: str | None, source: str, with_odds: bool, with_results: bool):
    """データを取得してDBに格納"""
    if target_date is None:
        target_date = date.today().isoformat()

    console.print(f"\n[bold blue]データ取得開始: {target_date} (ソース: {source})[/bold blue]\n")

    scraper = KdreamsScraper() if source == "kdreams" else WinticketScraper()
    session = get_session()

    try:
        # レース一覧を取得
        with console.status("レース一覧を取得中..."):
            races = scraper.scrape_race_list(target_date)

        if not races:
            console.print("[yellow]この日のレースが見つかりませんでした。[/yellow]")
            return

        console.print(f"[green]{len(races)}件のレースを検出[/green]\n")

        for i, race_info in enumerate(races):
            venue = race_info.get("venue_slug", "不明")
            rno = race_info.get("race_number", "?")
            console.print(f"  [{i+1}/{len(races)}] {venue} {rno}R ...")

            try:
                # 出走表を取得
                detail = scraper.scrape_race_detail(race_info)
                n_entries = len(detail.get("entries", []))

                venue_code = race_info.get("venue_code", "")
                race = store_race_entries(
                    session, target_date, venue_code, int(rno), detail,
                    race_name=race_info.get("race_name", ""),
                    grade=race_info.get("grade", ""),
                    venue_slug=race_info.get("venue_slug", ""),
                )

                if race and n_entries > 0:
                    console.print(f"    → [green]出走表: {n_entries}名[/green]")
                else:
                    console.print(f"    → [yellow]出走表: データなし[/yellow]")

                # オッズ取得
                if with_odds and race:
                    odds_list = scraper.scrape_odds(race_info)
                    if odds_list:
                        store_odds(session, race, odds_list)
                        console.print(f"    → [green]オッズ: {len(odds_list)}件[/green]")

                # 結果取得
                if with_results and race:
                    result_data = scraper.scrape_result(race_info)
                    if result_data.get("results"):
                        store_results(session, race, result_data)
                        console.print(f"    → [green]結果: {len(result_data['results'])}着[/green]")

            except Exception as e:
                session.rollback()
                console.print(f"    → [red]エラー: {e}[/red]")

        console.print(f"\n[bold green]完了![/bold green]")

    finally:
        session.close()


@cli.command("scrape-range")
@click.option("--from", "date_from", required=True, help="開始日 (YYYY-MM-DD)")
@click.option("--to", "date_to", default=None, help="終了日 (YYYY-MM-DD) デフォルト: 今日")
@click.option("--source", type=click.Choice(["winticket", "kdreams"]), default="kdreams", help="データソース")
@click.option("--with-results", is_flag=True, default=True, help="結果も取得する (デフォルト: ON)")
def scrape_range(date_from: str, date_to: str | None, source: str, with_results: bool):
    """期間指定で一括データ取得（学習データ蓄積用）"""
    from datetime import timedelta

    start = datetime.strptime(date_from, "%Y-%m-%d").date()
    end = datetime.strptime(date_to, "%Y-%m-%d").date() if date_to else date.today()

    if start > end:
        console.print("[red]開始日が終了日より後です[/red]")
        return

    total_days = (end - start).days + 1
    console.print(f"\n[bold blue]一括取得: {date_from} → {end.isoformat()} ({total_days}日間)[/bold blue]")
    console.print(f"ソース: {source} / 結果取得: {'ON' if with_results else 'OFF'}\n")

    scraper = KdreamsScraper() if source == "kdreams" else WinticketScraper()
    session = get_session()

    total_races = 0
    total_entries = 0
    total_results = 0
    error_days = []

    try:
        current = start
        day_num = 0
        while current <= end:
            day_num += 1
            current_str = current.isoformat()
            console.print(f"[bold]--- [{day_num}/{total_days}] {current_str} ---[/bold]")

            try:
                races = scraper.scrape_race_list(current_str)
                if not races:
                    console.print(f"  [dim]開催なし[/dim]")
                    current += timedelta(days=1)
                    continue

                console.print(f"  {len(races)}レース検出")

                day_entries = 0
                day_results = 0
                for race_info in races:
                    venue = race_info.get("venue_slug", "?")
                    rno = race_info.get("race_number", "?")

                    try:
                        # 出走表
                        detail = scraper.scrape_race_detail(race_info)
                        n_entries = len(detail.get("entries", []))

                        venue_code = race_info.get("venue_code", "")
                        race = store_race_entries(
                            session, current_str, venue_code, int(rno), detail,
                            race_name=race_info.get("race_name", ""),
                            grade=race_info.get("grade", ""),
                            venue_slug=race_info.get("venue_slug", ""),
                        )

                        day_entries += n_entries

                        # 結果取得
                        if with_results and race:
                            result_data = scraper.scrape_result(race_info)
                            if result_data.get("results"):
                                store_results(session, race, result_data)
                                day_results += len(result_data["results"])

                    except Exception as e:
                        session.rollback()
                        console.print(f"  [red]{venue} {rno}R: {e}[/red]")

                total_races += len(races)
                total_entries += day_entries
                total_results += day_results
                console.print(
                    f"  [green]→ {len(races)}R / {day_entries}名"
                    + (f" / 結果{day_results}件" if with_results else "")
                    + "[/green]"
                )

            except Exception as e:
                error_days.append(current_str)
                console.print(f"  [red]日単位エラー: {e}[/red]")

            current += timedelta(days=1)

        # サマリー
        console.print(f"\n[bold green]{'='*50}[/bold green]")
        console.print(f"[bold green]完了![/bold green]")
        console.print(f"  期間: {date_from} → {end.isoformat()} ({total_days}日)")
        console.print(f"  レース: {total_races}")
        console.print(f"  出走エントリー: {total_entries}")
        if with_results:
            console.print(f"  結果データ: {total_results}")
        if error_days:
            console.print(f"  [yellow]エラー日: {', '.join(error_days)}[/yellow]")

    finally:
        session.close()


@cli.command()
@click.option("--date", "target_date", default=None, help="対象日 (YYYY-MM-DD)")
@click.option("--venue", default=None, help="競輪場名でフィルタ")
@click.option("--race", "race_number", default=None, type=int, help="レース番号")
@click.option("--mode", type=click.Choice(["accuracy", "roi"]), default="accuracy", help="予想モード")
@click.option("--budget", default=1000, type=int, help="予算(円)")
def predict(target_date: str | None, venue: str | None, race_number: int | None,
            mode: str, budget: int):
    """レースの予想を表示"""
    if target_date is None:
        target_date = date.today().isoformat()

    session = get_session()
    predictor = KeirinPredictor(mode=mode)

    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        query = session.query(Race).filter(Race.race_date == dt)

        if venue:
            rc = session.query(Racecourse).filter(Racecourse.name.like(f"%{venue}%")).first()
            if rc:
                query = query.filter(Race.racecourse_id == rc.id)
            else:
                console.print(f"[red]競輪場 '{venue}' が見つかりません[/red]")
                return

        if race_number:
            query = query.filter(Race.race_number == race_number)

        races = query.order_by(Race.racecourse_id, Race.race_number).all()

        if not races:
            console.print(f"[yellow]{target_date} のレースデータがありません。先にscrapeを実行してください。[/yellow]")
            return

        mode_label = "的中率重視" if mode == "accuracy" else "回収率重視"
        console.print(f"\n[bold blue]予想モード: {mode_label}[/bold blue]\n")

        for race in races:
            racecourse = session.query(Racecourse).filter_by(id=race.racecourse_id).first()
            venue_name = racecourse.name if racecourse else "不明"

            # 予想実行
            predictions = predictor.predict(session, race)
            if not predictions:
                continue

            # レースヘッダー
            header = f"{venue_name} {race.race_number}R"
            if race.race_name:
                header += f"  {race.race_name}"
            if race.grade:
                header += f"  [{race.grade}]"

            console.print(Panel(f"[bold]{header}[/bold]", style="blue"))

            # 予想テーブル
            table = Table(show_header=True, header_style="bold cyan", show_lines=False)
            table.add_column("予想", width=8, no_wrap=True)
            table.add_column("車番", justify="center", width=4, no_wrap=True)
            table.add_column("選手名", width=10, no_wrap=True)
            table.add_column("級", justify="center", width=3, no_wrap=True)
            table.add_column("勝率", justify="right", width=5, no_wrap=True)
            table.add_column("2連率", justify="right", width=5, no_wrap=True)
            table.add_column("3連率", justify="right", width=5, no_wrap=True)
            table.add_column("スコア", justify="right", width=7, no_wrap=True)
            if mode == "roi":
                table.add_column("ｵｯｽﾞ", justify="right", width=6, no_wrap=True)
                table.add_column("期待値", justify="right", width=6, no_wrap=True)

            for p in predictions:
                rec = p["recommendation"]
                style = ""
                if "◎" in rec:
                    style = "bold red"
                elif "○" in rec:
                    style = "bold yellow"
                elif "▲" in rec:
                    style = "blue"

                second_rate = p.get("second_rate", 0) if isinstance(p.get("second_rate"), (int, float)) else 0
                third_rate = p.get("third_rate", 0) if isinstance(p.get("third_rate"), (int, float)) else 0

                row = [
                    rec,
                    str(p["car_number"]),
                    p["player_name"],
                    p.get("grade", ""),
                    f"{p['win_rate']:.1f}" if p["win_rate"] else "-",
                    f"{second_rate:.1f}" if second_rate else "-",
                    f"{third_rate:.1f}" if third_rate else "-",
                    f"{p['score']:.2f}",
                ]
                if mode == "roi":
                    row.append(f"{p['odds']:.1f}" if p.get("odds") else "-")
                    row.append(f"{p['expected_value']:.2f}" if p.get("expected_value") else "-")

                table.add_row(*row, style=style)

            console.print(table)

            # 買い目提案
            bets = predictor.suggest_bets(predictions, budget)
            if bets:
                console.print(f"\n  [bold]推奨買い目 (予算 {budget:,}円):[/bold]")
                for bet in bets:
                    console.print(
                        f"    {bet['bet_type']} {bet['combination']}  "
                        f"{bet['amount']:,}円  ← {bet['reason']}"
                    )
            console.print()

    finally:
        session.close()


@cli.command()
@click.option("--mode", type=click.Choice(["accuracy", "roi"]), default="accuracy")
@click.option("--min-races", default=50, type=int, help="最低学習レース数")
def train(mode: str, min_races: int):
    """モデルを学習"""
    session = get_session()
    predictor = KeirinPredictor(mode=mode)

    mode_label = "的中率重視" if mode == "accuracy" else "回収率重視"
    console.print(f"\n[bold blue]モデル学習: {mode_label}[/bold blue]\n")

    with console.status("学習中..."):
        result = predictor.train(session, min_races=min_races)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
    else:
        console.print(f"[green]学習完了![/green]")
        console.print(f"  レース数: {result['n_races']}")
        console.print(f"  サンプル数: {result['n_samples']}")
        console.print(f"  CV Score: {result['cv_score']:.4f} (±{result['cv_std']:.4f})")

    session.close()


@cli.command()
def status():
    """DB状況を表示"""
    session = get_session()

    races_total = session.query(Race).count()
    races_finished = session.query(Race).filter(Race.status == "finished").count()
    entries_total = session.query(RaceEntry).count()
    entries_with_result = session.query(RaceEntry).filter(RaceEntry.finish_position.isnot(None)).count()

    from src.common.database import Player, Odds
    players_total = session.query(Player).count()
    odds_total = session.query(Odds).count()

    table = Table(title="データベース状況", show_header=True, header_style="bold cyan")
    table.add_column("項目", width=20)
    table.add_column("件数", justify="right", width=10)

    table.add_row("レース (全体)", f"{races_total:,}")
    table.add_row("レース (確定)", f"{races_finished:,}")
    table.add_row("出走エントリー", f"{entries_total:,}")
    table.add_row("結果確定エントリー", f"{entries_with_result:,}")
    table.add_row("登録選手", f"{players_total:,}")
    table.add_row("オッズデータ", f"{odds_total:,}")

    # モデル状態
    from src.common.config import PROJECT_ROOT
    accuracy_model = (PROJECT_ROOT / "data" / "models" / "model_accuracy.pkl").exists()
    roi_model = (PROJECT_ROOT / "data" / "models" / "model_roi.pkl").exists()
    table.add_row("", "")
    table.add_row("的中率モデル", "[green]あり[/green]" if accuracy_model else "[yellow]なし[/yellow]")
    table.add_row("回収率モデル", "[green]あり[/green]" if roi_model else "[yellow]なし[/yellow]")

    console.print()
    console.print(table)
    console.print()

    session.close()


if __name__ == "__main__":
    cli()
