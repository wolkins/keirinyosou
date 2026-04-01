"""データベースモデル定義"""
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from .config import DB_URL


class Base(DeclarativeBase):
    pass


class Racecourse(Base):
    """競輪場マスタ"""
    __tablename__ = "racecourses"

    id = Column(Integer, primary_key=True)
    code = Column(String(4), unique=True, nullable=False)  # 競輪場コード
    name = Column(String(50), nullable=False)               # 競輪場名
    prefecture = Column(String(20))                          # 都道府県
    track_length = Column(Integer)                           # バンク周長(m): 333, 400, 500

    races = relationship("Race", back_populates="racecourse")


class Player(Base):
    """選手マスタ"""
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    player_id = Column(String(10), unique=True, nullable=False)  # 選手登録番号
    name = Column(String(50), nullable=False)
    name_kana = Column(String(100))
    gender = Column(String(2))                # M/F
    prefecture = Column(String(20))           # 登録地
    grade = Column(String(4))                 # 級班: SS, S1, S2, A1, A2, A3
    age = Column(Integer)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    entries = relationship("RaceEntry", back_populates="player")


class Race(Base):
    """レース情報"""
    __tablename__ = "races"

    id = Column(Integer, primary_key=True)
    race_date = Column(Date, nullable=False)
    racecourse_id = Column(Integer, ForeignKey("racecourses.id"), nullable=False)
    race_number = Column(Integer, nullable=False)           # R番号 (1-12)
    race_name = Column(String(100))                          # レース名
    grade = Column(String(10))                               # GP, GI, GII, GIII, FI, FII
    round_name = Column(String(50))                          # 初日, 2日目, 決勝 等
    distance = Column(Integer)                               # 距離(m)
    is_girl = Column(Boolean, default=False)                 # ガールズケイリン
    status = Column(String(20), default="scheduled")         # scheduled, finished, cancelled

    racecourse = relationship("Racecourse", back_populates="races")
    entries = relationship("RaceEntry", back_populates="race")
    odds = relationship("Odds", back_populates="race")

    __table_args__ = (
        UniqueConstraint("race_date", "racecourse_id", "race_number", name="uq_race"),
    )


class RaceEntry(Base):
    """出走表（各レースの選手エントリー）"""
    __tablename__ = "race_entries"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    frame_number = Column(Integer, nullable=False)           # 枠番 (1-9)
    car_number = Column(Integer, nullable=False)             # 車番 (1-9)
    line_group = Column(String(20))                          # ライン (先行, 番手, 3番手等)
    line_label = Column(String(50))                          # ライン名 (北日本, 関東 等)

    # 直近成績
    win_rate = Column(Float)          # 勝率
    second_rate = Column(Float)       # 2連対率
    third_rate = Column(Float)        # 3連対率
    avg_start = Column(Float)         # 平均ST

    # 選手コメント
    comment = Column(Text)

    # レース結果
    finish_position = Column(Integer)  # 着順 (NULL=未確定)
    finish_time = Column(String(10))   # ゴールタイム
    win_technique = Column(String(20)) # 決まり手 (逃げ, 捲り, 差し, マーク)

    race = relationship("Race", back_populates="entries")
    player = relationship("Player", back_populates="entries")

    __table_args__ = (
        UniqueConstraint("race_id", "car_number", name="uq_entry"),
    )


class Odds(Base):
    """オッズ"""
    __tablename__ = "odds"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    bet_type = Column(String(20), nullable=False)  # 2car_quinella, 2car_exacta, 3car_quinella, 3car_exacta, win, place
    combination = Column(String(20), nullable=False)  # "1-2", "1-2-3" 等
    odds_value = Column(Float, nullable=False)
    captured_at = Column(DateTime, default=datetime.now)

    race = relationship("Race", back_populates="odds")

    __table_args__ = (
        UniqueConstraint("race_id", "bet_type", "combination", name="uq_odds"),
    )


class PredictionResult(Base):
    """予想結果の記録"""
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    mode = Column(String(20), nullable=False)        # accuracy (的中率) / roi (回収率)
    bet_type = Column(String(20), nullable=False)     # 賭式
    combination = Column(String(20), nullable=False)  # 予想組み合わせ
    confidence = Column(Float)                         # 信頼度スコア
    expected_value = Column(Float)                     # 期待値
    is_hit = Column(Boolean)                           # 的中したか (結果確定後)
    payout = Column(Float)                             # 払戻金
    created_at = Column(DateTime, default=datetime.now)


# エンジン・セッション
engine = create_engine(DB_URL, echo=False)


def init_db():
    """テーブル作成"""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """セッション取得"""
    return sessionmaker(bind=engine)()


# 競輪場マスタデータ（公式JKAコード）
RACECOURSES = [
    {"code": "11", "name": "函館", "prefecture": "北海道", "track_length": 400},
    {"code": "12", "name": "青森", "prefecture": "青森県", "track_length": 400},
    {"code": "13", "name": "いわき平", "prefecture": "福島県", "track_length": 400},
    {"code": "21", "name": "弥彦", "prefecture": "新潟県", "track_length": 400},
    {"code": "22", "name": "前橋", "prefecture": "群馬県", "track_length": 335},
    {"code": "23", "name": "取手", "prefecture": "茨城県", "track_length": 400},
    {"code": "24", "name": "宇都宮", "prefecture": "栃木県", "track_length": 500},
    {"code": "25", "name": "大宮", "prefecture": "埼玉県", "track_length": 500},
    {"code": "26", "name": "西武園", "prefecture": "埼玉県", "track_length": 400},
    {"code": "27", "name": "京王閣", "prefecture": "東京都", "track_length": 400},
    {"code": "28", "name": "立川", "prefecture": "東京都", "track_length": 400},
    {"code": "31", "name": "松戸", "prefecture": "千葉県", "track_length": 333},
    {"code": "32", "name": "千葉", "prefecture": "千葉県", "track_length": 500},
    {"code": "33", "name": "川崎", "prefecture": "神奈川県", "track_length": 400},
    {"code": "34", "name": "平塚", "prefecture": "神奈川県", "track_length": 400},
    {"code": "35", "name": "小田原", "prefecture": "神奈川県", "track_length": 333},
    {"code": "36", "name": "伊東温泉", "prefecture": "静岡県", "track_length": 333},
    {"code": "37", "name": "静岡", "prefecture": "静岡県", "track_length": 400},
    {"code": "38", "name": "名古屋", "prefecture": "愛知県", "track_length": 400},
    {"code": "42", "name": "名古屋", "prefecture": "愛知県", "track_length": 400},
    {"code": "43", "name": "岐阜", "prefecture": "岐阜県", "track_length": 400},
    {"code": "44", "name": "大垣", "prefecture": "岐阜県", "track_length": 400},
    {"code": "45", "name": "豊橋", "prefecture": "愛知県", "track_length": 400},
    {"code": "46", "name": "富山", "prefecture": "富山県", "track_length": 333},
    {"code": "47", "name": "松阪", "prefecture": "三重県", "track_length": 400},
    {"code": "48", "name": "四日市", "prefecture": "三重県", "track_length": 400},
    {"code": "51", "name": "福井", "prefecture": "福井県", "track_length": 400},
    {"code": "53", "name": "奈良", "prefecture": "奈良県", "track_length": 333},
    {"code": "54", "name": "向日町", "prefecture": "京都府", "track_length": 400},
    {"code": "55", "name": "和歌山", "prefecture": "和歌山県", "track_length": 400},
    {"code": "56", "name": "岸和田", "prefecture": "大阪府", "track_length": 400},
    {"code": "61", "name": "玉野", "prefecture": "岡山県", "track_length": 400},
    {"code": "62", "name": "広島", "prefecture": "広島県", "track_length": 400},
    {"code": "63", "name": "防府", "prefecture": "山口県", "track_length": 333},
    {"code": "71", "name": "高松", "prefecture": "香川県", "track_length": 400},
    {"code": "72", "name": "小松島", "prefecture": "徳島県", "track_length": 400},
    {"code": "73", "name": "高知", "prefecture": "高知県", "track_length": 500},
    {"code": "74", "name": "松山", "prefecture": "愛媛県", "track_length": 400},
    {"code": "81", "name": "小倉", "prefecture": "福岡県", "track_length": 400},
    {"code": "82", "name": "久留米", "prefecture": "福岡県", "track_length": 400},
    {"code": "83", "name": "武雄", "prefecture": "佐賀県", "track_length": 400},
    {"code": "84", "name": "佐世保", "prefecture": "長崎県", "track_length": 400},
    {"code": "85", "name": "別府", "prefecture": "大分県", "track_length": 400},
    {"code": "86", "name": "熊本", "prefecture": "熊本県", "track_length": 400},
]


def seed_racecourses():
    """競輪場マスタの初期投入"""
    session = get_session()
    try:
        for rc in RACECOURSES:
            existing = session.query(Racecourse).filter_by(code=rc["code"]).first()
            if not existing:
                session.add(Racecourse(**rc))
        session.commit()
    finally:
        session.close()
