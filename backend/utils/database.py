import pymysql
import config
import logging
from datetime import datetime
import time
from typing import Dict, List

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, max_retries=3, retry_delay=5):
        """Initialize database configuration and connection."""
        self.config = {
            'host': config.config["MYSQL_HOST"],
            'user': config.config["MYSQL_USER"],
            'password': config.config["MYSQL_PASSWORD"],
            'database': config.config["MYSQL_DATABASE"],
            'port': config.config["MYSQL_PORT"],
            'charset': 'utf8mb4',
            'connect_timeout': 30
        }
        if not isinstance(self.config['port'], int):
            logger.error(f"Invalid MySQL port: {self.config['port']} (must be an integer)")
            raise ValueError(f"Invalid MySQL port: {self.config['port']}")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection = None

    def connect(self):
        """Establish a database connection with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.connection = pymysql.connect(**self.config)
                logger.info("Successfully connected to database")
                return
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        logger.error("Failed to connect to database after all retries")
        self.connection = None

    def init_db(self):
        """Initialize the database and create necessary tables."""
        # Connect without specifying database to create it if it doesn't exist
        temp_config = self.config.copy()
        temp_config.pop('database')
        conn = None
        try:
            conn = pymysql.connect(**temp_config)
            with conn.cursor() as c:
                c.execute(f"CREATE DATABASE IF NOT EXISTS {self.config['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                conn.commit()
            logger.info(f"Created database {self.config['database']} if it didn't exist")
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
        finally:
            if conn:
                conn.close()

        # Connect to the database and create tables
        if not self.connection:
            self.connect()
        if not self.connection:
            logger.error("Failed to initialize database due to connection error")
            raise RuntimeError("Database connection failed")

        try:
            with self.connection.cursor() as c:
                # Create signals table
                c.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(50) NOT NULL,
                        timeframe VARCHAR(10) NOT NULL,
                        timestamp DATETIME NOT NULL,
                        trade_signal VARCHAR(20) NOT NULL,
                        signal_type VARCHAR(20) NOT NULL,
                        current_price DECIMAL(10,5) NOT NULL,
                        entry_level DECIMAL(10,5) NOT NULL,
                        stop_loss DECIMAL(10,5) NOT NULL,
                        take_profit DECIMAL(10,5) NOT NULL,
                        risk_reward_ratio DECIMAL(5,2) NOT NULL,
                        confidence DECIMAL(5,2) NOT NULL,
                        atr DECIMAL(10,5) NOT NULL,
                        status ENUM('Open', 'Closed - TP Hit', 'Closed - SL Hit', 'Closed - Expired', 'Closed - Manual') NOT NULL DEFAULT 'Open',
                        close_time DATETIME,
                        close_price DECIMAL(10,5),
                        pips_gained DECIMAL(10,2),
                        duration_minutes INT,
                        comments TEXT,
                        INDEX idx_symbol (symbol),
                        INDEX idx_timestamp (timestamp),
                        INDEX idx_status (status),
                        INDEX idx_symbol_status (symbol, status)
                    ) ENGINE=InnoDB
                ''')

                # Drop the procedure if it exists first to avoid conflicts
                c.execute("DROP PROCEDURE IF EXISTS check_expired_signals")
                
                # Create stored procedure for checking expired signals with corrected syntax
                c.execute('''
                    CREATE PROCEDURE check_expired_signals()
                    BEGIN
                        DECLARE expiry_hours INT;
                        
                        UPDATE signals 
                        SET status = 'Closed - Expired',
                            close_time = NOW(),
                            close_price = current_price,
                            duration_minutes = TIMESTAMPDIFF(MINUTE, timestamp, NOW())
                        WHERE status = 'Open'
                        AND (
                            (signal_type = 'Long-Term' AND timestamp < NOW() - INTERVAL 24 HOUR) OR
                            (signal_type = 'Short-Term' AND timestamp < NOW() - INTERVAL 4 HOUR)
                        );
                    END
                ''')

                self.connection.commit()
            logger.info("Database schema initialized with signals table and stored procedure")
        except Exception as e:
            logger.error(f"Error creating table or procedure: {e}")
            raise

    def save_signal(self, signal: Dict):
        """Save a trading signal to the database."""
        if not self.connection:
            self.connect()
        if not self.connection:
            logger.error("Cannot save signal due to database connection failure")
            return

        try:
            with self.connection.cursor() as c:
                c.execute('''
                    INSERT INTO signals (
                        symbol, timeframe, timestamp, trade_signal, signal_type, current_price,
                        entry_level, stop_loss, take_profit, risk_reward_ratio, confidence,
                        atr, status, comments
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    signal['symbol'], signal['timeframe'], signal['timestamp'],
                    signal['trade_signal'], signal['signal_type'], signal['current_price'],
                    signal['entry_level'], signal['stop_loss'], signal['take_profit'],
                    signal['risk_reward_ratio'], signal['confidence'], signal['atr'],
                    'Open', signal.get('comments', '')
                ))
                self.connection.commit()
            logger.info(f"Saved signal for {signal['symbol']} at {signal['timestamp']}")
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            try:
                with open('logs/signals_fallback.log', 'a') as f:
                    f.write(f"{datetime.now()}: {signal}\n")
                logger.info(f"Saved signal to fallback log for {signal['symbol']}")
            except Exception as e:
                logger.error(f"Failed to save signal to fallback log: {e}")

    def update_signal_status(self, signal_id: int, new_status: str, close_price: float = None) -> bool:
        """Update signal status with transaction safety."""
        if not self.connection:
            self.connect()
        if not self.connection:
            logger.error("Failed to update signal status due to connection error")
            return False

        try:
            with self.connection.cursor() as c:
                update_query = '''
                    UPDATE signals 
                    SET status = %s,
                        close_time = %s,
                        close_price = %s,
                        pips_gained = CASE
                            WHEN trade_signal = 'BUY' THEN (%s - entry_level) * 10000
                            WHEN trade_signal = 'SELL' THEN (entry_level - %s) * 10000
                            ELSE NULL
                        END,
                        duration_minutes = TIMESTAMPDIFF(MINUTE, timestamp, %s)
                    WHERE id = %s
                '''
                close_time = datetime.now()
                c.execute(update_query, (
                    new_status, close_time, close_price,
                    close_price if close_price is not None else 0,
                    close_price if close_price is not None else 0,
                    close_time, signal_id
                ))
                self.connection.commit()
                if c.rowcount > 0:
                    logger.info(f"Updated signal {signal_id} to status {new_status}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error updating signal status: {e}")
            return False

    def get_open_signals(self, symbol: str = None, timeframe: str = None) -> List[Dict]:
        """Retrieve open signals, optionally filtered by symbol and timeframe."""
        if not self.connection:
            self.connect()
        if not self.connection:
            logger.error("Failed to fetch open signals due to connection error")
            return []

        try:
            with self.connection.cursor() as c:
                query = "SELECT * FROM signals WHERE status = 'Open'"
                params = []
                if symbol:
                    query += " AND symbol = %s"
                    params.append(symbol)
                if timeframe:
                    query += " AND timeframe = %s"
                    params.append(timeframe)
                query += " ORDER BY timestamp DESC"
                c.execute(query, params)
                columns = [col[0] for col in c.description]
                signals = [dict(zip(columns, row)) for row in c.fetchall()]
            #logger.info(f"Retrieved {len(signals)} open signals")
            return signals
        except Exception as e:
            logger.error(f"Error fetching open signals: {e}")
            return []

    def get_recent_signals(self, limit: int = 5) -> List[Dict]:
        """Retrieve recent signals for display."""
        if not self.connection:
            self.connect()
        if not self.connection:
            logger.error("Failed to fetch recent signals due to connection error")
            return []

        try:
            with self.connection.cursor() as c:
                c.execute('''
                    SELECT id, symbol, timeframe, trade_signal, signal_type, status, 
                           current_price, entry_level, stop_loss, take_profit, 
                           pips_gained, timestamp
                    FROM signals
                    ORDER BY timestamp DESC
                    LIMIT %s
                ''', (limit,))
                signals = [
                    {
                        "id": r[0], "symbol": r[1], "timeframe": r[2], 
                        "trade_signal": r[3], "signal_type": r[4], "status": r[5],
                        "current_price": float(r[6]) if r[6] is not None else 0,
                        "entry_level": float(r[7]) if r[7] is not None else 0,
                        "stop_loss": float(r[8]) if r[8] is not None else 0,
                        "take_profit": float(r[9]) if r[9] is not None else 0,
                        "pips_gained": float(r[10]) if r[10] is not None else 0,
                        "timestamp": r[11]
                    }
                    for r in c.fetchall()
                ]
            #logger.info(f"Retrieved {len(signals)} recent signals")
            return signals
        except Exception as e:
            logger.error(f"Error fetching recent signals: {e}")
            return []

    def get_signal_stats(self, limit: int = 100) -> Dict:
        """Retrieve performance statistics for signals."""
        if not self.connection:
            self.connect()
        if not self.connection:
            logger.error("Failed to fetch signal stats due to connection error")
            return {
                "long_term_win_rate": 0, 
                "short_term_win_rate": 0, 
                "total_signals": 0, 
                "buy_signals": 0, 
                "sell_signals": 0
            }

        try:
            with self.connection.cursor() as c:
                query = """
                    SELECT 
                        (SELECT COUNT(*) FROM signals) as total_signals,
                        SUM(CASE WHEN status = %s THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN status = %s THEN 1 ELSE 0 END) as losses,
                        AVG(CASE WHEN status LIKE %s THEN pips_gained ELSE NULL END) as avg_pips,
                        AVG(risk_reward_ratio) as avg_rr,
                        SUM(CASE WHEN status = %s AND signal_type = %s THEN 1 ELSE 0 END) /
                            NULLIF(SUM(CASE WHEN signal_type = %s AND status LIKE %s THEN 1 ELSE 0 END), 0) * 100 as long_term_win_rate,
                        SUM(CASE WHEN status = %s AND signal_type = %s THEN 1 ELSE 0 END) /
                            NULLIF(SUM(CASE WHEN signal_type = %s AND status LIKE %s THEN 1 ELSE 0 END), 0) * 100 as short_term_win_rate,
                        SUM(CASE WHEN trade_signal = %s AND status LIKE %s THEN 1 ELSE 0 END) as buy_signals,
                        SUM(CASE WHEN trade_signal = %s AND status LIKE %s THEN 1 ELSE 0 END) as sell_signals
                    FROM signals
                    WHERE status LIKE %s
                    LIMIT %s
                """
                params = (
                    'Closed - TP Hit',  # wins
                    'Closed - SL Hit',  # losses
                    'Closed%',          # avg_pips
                    'Closed - TP Hit',  # long_term_win_rate
                    'Long-Term',        # long_term_win_rate
                    'Long-Term',        # long_term_win_rate
                    'Closed%',          # long_term_win_rate
                    'Closed - TP Hit',  # short_term_win_rate
                    'Short-Term',       # short_term_win_rate
                    'Short-Term',       # short_term_win_rate
                    'Closed%',          # short_term_win_rate
                    'BUY',              # buy_signals
                    'Closed%',          # buy_signals
                    'SELL',             # sell_signals
                    'Closed%',          # sell_signals
                    'Closed%',          # WHERE clause
                    limit               # LIMIT clause
                )
                c.execute(query, params)
                result = c.fetchone()
                if not result:
                    logger.warning("No closed signals found in database")
                    return {
                        "long_term_win_rate": 0, 
                        "short_term_win_rate": 0, 
                        "total_signals": 0, 
                        "buy_signals": 0, 
                        "sell_signals": 0
                    }
                stats = {
                    "total_signals": result[0] if result[0] is not None else 0,
                    "wins": result[1] if result[1] is not None else 0,
                    "losses": result[2] if result[2] is not None else 0,
                    "avg_pips": float(result[3]) if result[3] else 0,
                    "avg_rr": float(result[4]) if result[4] else 0,
                    "long_term_win_rate": float(result[5]) if result[5] else 0,
                    "short_term_win_rate": float(result[6]) if result[6] else 0,
                    "buy_signals": result[7] if result[7] is not None else 0,
                    "sell_signals": result[8] if result[8] is not None else 0
                }
                #logger.info(f"Retrieved signal stats: {stats}")
                return stats
        except Exception as e:
            logger.error(f"Error fetching signal stats: {e}")
            return {
                "long_term_win_rate": 0, 
                "short_term_win_rate": 0, 
                "total_signals": 0, 
                "buy_signals": 0, 
                "sell_signals": 0
            }

    def execute_stored_procedure(self, procedure_name: str, params: tuple = ()) -> bool:
        """Execute a stored procedure with given parameters."""
        if not self.connection:
            self.connect()
        if not self.connection:
            logger.error(f"Failed to execute stored procedure {procedure_name} due to connection error")
            return False

        try:
            with self.connection.cursor() as c:
                c.callproc(procedure_name, params)
                self.connection.commit()
            logger.info(f"Executed stored procedure {procedure_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to execute stored procedure {procedure_name}: {e}")
            return False

    def __del__(self):
        """Close the database connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
                