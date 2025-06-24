import mysql.connector
from datetime import datetime

from config import HOST, USER, PASS, DB, TABLE1

def get_db():
    return mysql.connector.connect(
        host=HOST,
        user=USER,
        password=PASS,
        database=DB
    )

def format_local_time(timestamp_val):
    """Convert UNIX timestamp to human-readable local time string."""
    try:
        ts_float = float(timestamp_val)
        return datetime.fromtimestamp(ts_float).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(timestamp_val)

def format_prediction_data(records):
    """Format the database records into the required JSON structure."""
    formatted_data = []
    for row in records:
        formatted_row = {
            "timestamp": format_local_time(row["timestamp"]),
            "source_ip": str(row["source_ip"]),
            "destination_ip": str(row["destination_ip"]),
            "protocol": str(row["protocol"]),
            "packet_size": str(row["packet_size"]),
            "src_port": str(row["src_port"]),
            "dst_port": str(row["dst_port"]),
            "tcp_flags": str(row["tcp_flags"]),
            "ttl": str(row["ttl"]),
            "http_host": str(row["http_host"]),
            "http_uri": "*",  # As per your example
            "tls_sni": str(row["tls_sni"]),
            "full_url": str(row["full_url"]),
            "prediction": str(row["prediction"]).upper()  # Ensures uppercase label
        }
        formatted_data.append(formatted_row)

    return formatted_data

def get_predictions(page: int, page_size: int):
    """Query the database with pagination and return the formatted predictions."""
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        
        # Calculate OFFSET based on page number and page size
        offset = (page - 1) * page_size
        
        # Fetch most recent entries first using ORDER BY (descending)
        cursor.execute(f"""
            SELECT * FROM {TABLE1}
            ORDER BY id DESC
            LIMIT {page_size} OFFSET {offset}
        """)
        records = cursor.fetchall()

        # Format the records as needed
        formatted_predictions = format_prediction_data(records)
        
        cursor.close()
        db.close()
        
        return formatted_predictions
    except Exception as e:
        return {"error": f"Failed to fetch data from database: {str(e)}"}
