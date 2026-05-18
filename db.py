import os
import psycopg2

def get_conn():
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(url)

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    order_id         INTEGER PRIMARY KEY,
                    accept_time      TIMESTAMP,
                    accept_gps_lng   DOUBLE PRECISION,
                    accept_gps_lat   DOUBLE PRECISION,
                    delivery_gps_lng DOUBLE PRECISION,
                    delivery_gps_lat DOUBLE PRECISION,
                    stop_package_count INTEGER,
                    stop_index       INTEGER,
                    remaining_stops  INTEGER,
                    segment_distance DOUBLE PRECISION,
                    predicted_min    DOUBLE PRECISION,
                    eta              TEXT,
                    delivery_time    TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS delivery_log (
                    id               SERIAL PRIMARY KEY,
                    order_id         INTEGER,
                    courier_id       TEXT,
                    accept_time      TIMESTAMP,
                    accept_gps_lng   DOUBLE PRECISION,
                    accept_gps_lat   DOUBLE PRECISION,
                    delivery_gps_lng DOUBLE PRECISION,
                    delivery_gps_lat DOUBLE PRECISION,
                    stop_package_count INTEGER,
                    predicted_minutes  DOUBLE PRECISION,
                    eta              TEXT,
                    delivery_time    TIMESTAMP
                )
            """)
        conn.commit()
