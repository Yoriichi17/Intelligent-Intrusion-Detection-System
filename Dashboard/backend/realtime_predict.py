import pandas as pd
import joblib
import time
import os
import sys
import mysql.connector
from threading import Lock
from config import HOST, USER, PASS, DB, TABLE1

sys.stdout.reconfigure(encoding='utf-8')

# Paths
INPUT_CSV = 'capture_data.csv'
OUTPUT_CSV = 'output_predictions.csv'
model_path = 'attack_detector_model.pkl'
label_path = 'label_encoder.pkl'
ip_path = 'ip_encoder.pkl'
flags_path = 'flags_encoder.pkl'
http_host_enc_path = 'http_host_encoder.pkl'
http_uri_enc_path = 'http_uri_encoder.pkl'
tls_sni_enc_path = 'tls_sni_encoder.pkl'

# Features used for prediction
features = ['source_ip', 'destination_ip', 'protocol', 'packet_size',
            'src_port', 'dst_port', 'tcp_flags', 'ttl',
            'http_host', 'http_uri', 'tls_sni']

essential_cols = ['timestamp', 'source_ip', 'destination_ip', 'protocol',
                  'packet_size', 'src_port', 'dst_port', 'ttl']

safe_domains = ['youtube.com', 'instagram.com', 'facebook.com', 'x.com', 'twitter.com']

model_mtime = None
lock = Lock()
model = label_encoder = ip_encoder = flags_encoder = None
http_host_encoder = http_uri_encoder = tls_sni_encoder = None

def load_models():
    global model, label_encoder, ip_encoder, flags_encoder
    global http_host_encoder, http_uri_encoder, tls_sni_encoder, model_mtime
    with lock:
        current_mtime = os.path.getmtime(model_path)
        if model_mtime != current_mtime:
            print("🔁 Detected updated model, reloading...")
            model = joblib.load(model_path)
            label_encoder = joblib.load(label_path)
            ip_encoder = joblib.load(ip_path)
            flags_encoder = joblib.load(flags_path)
            http_host_encoder = joblib.load(http_host_enc_path)
            http_uri_encoder = joblib.load(http_uri_enc_path)
            tls_sni_encoder = joblib.load(tls_sni_enc_path)
            model_mtime = current_mtime
            print("Model and encoders loaded.")

# Initialize output CSV with headers if not exists
if not os.path.exists(OUTPUT_CSV):
    pd.DataFrame(columns=['timestamp', 'source_ip_raw', 'destination_ip_raw', 'http_host', 'http_uri', 'tls_sni', 'full_url'] + features + ['Prediction'])\
      .to_csv(OUTPUT_CSV, index=False)

def get_db():
    return mysql.connector.connect(
        host=HOST,
        user=USER,
        password=PASS,
        database=DB
    )

def safe_int(col, df):
    # Safely convert to int, replacing invalid with 0
    return pd.to_numeric(df[col].astype(str).str.split('#').str[0].str.strip().replace('', '0'), errors='coerce').fillna(0).astype(int)

def insert_into_db(data):
    try:
        db = get_db()
        cursor = db.cursor()

        for _, row in data.iterrows():
            row = row.fillna('')
            sql = f"""INSERT INTO {TABLE1} (timestamp, source_ip, destination_ip, protocol, packet_size, 
                     src_port, dst_port, tcp_flags, ttl, http_host, http_uri, tls_sni, full_url, prediction) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (
                row['timestamp'], row['source_ip_raw'], row['destination_ip_raw'], row['protocol'], row['packet_size'],
                row['src_port'], row['dst_port'], row['tcp_flags'], row['ttl'], row['http_host'], row.get('http_uri', ''),
                row['tls_sni'], row.get('full_url', ''), row['Prediction']
            ))

        db.commit()
        cursor.close()
        db.close()
    except Exception as e:
        print("⚠ Error in DB insertion:", e)

def process_new_data():
    try:
        # Read number of rows already processed
        processed_rows = pd.read_csv(OUTPUT_CSV, low_memory=False).shape[0]
        data = pd.read_csv(INPUT_CSV, low_memory=False)

        if data.shape[0] <= processed_rows:
            # No new data to process
            return

        new_data = data.iloc[processed_rows:]
        new_data = new_data.dropna(subset=essential_cols)
        if new_data.empty:
            return

        # Convert numeric columns safely
        for col in ['packet_size', 'src_port', 'dst_port', 'ttl', 'protocol']:
            new_data[col] = safe_int(col, new_data)

        # Fill missing IPs with '0' string before encoding
        new_data['source_ip'] = new_data['source_ip'].fillna('0').astype(str)
        new_data['destination_ip'] = new_data['destination_ip'].fillna('0').astype(str)

        new_data['source_ip_raw'] = new_data['source_ip']
        new_data['destination_ip_raw'] = new_data['destination_ip']

        # Reload model and encoders if updated
        load_models()

        # Encode IP addresses - handle unseen IPs as 0
        ip_classes = ip_encoder.classes_.tolist()
        new_data['source_ip'] = new_data['source_ip'].apply(
            lambda ip: ip_encoder.transform([ip])[0] if ip in ip_classes else 0)
        new_data['destination_ip'] = new_data['destination_ip'].apply(
            lambda ip: ip_encoder.transform([ip])[0] if ip in ip_classes else 0)

        # Encode tcp_flags - fill missing with '0'
        if 'tcp_flags' in new_data.columns:
            flags_classes = flags_encoder.classes_.tolist()
            new_data['tcp_flags'] = new_data['tcp_flags'].fillna('0').astype(str).apply(
                lambda flag: flags_encoder.transform([flag])[0] if flag in flags_classes else 0)
        else:
            new_data['tcp_flags'] = 0

        # Encode http_host
        http_host_classes = http_host_encoder.classes_.tolist()
        new_data['http_host'] = new_data['http_host'].fillna('unknown').astype(str).apply(
            lambda x: http_host_encoder.transform([x])[0] if x in http_host_classes else 0)

        # Encode http_uri
        if 'http_uri' in new_data.columns:
            http_uri_classes = http_uri_encoder.classes_.tolist()
            new_data['http_uri'] = new_data['http_uri'].fillna('unknown').astype(str).apply(
                lambda x: http_uri_encoder.transform([x])[0] if x in http_uri_classes else 0)
        else:
            new_data['http_uri'] = 0

        # Encode tls_sni
        tls_sni_classes = tls_sni_encoder.classes_.tolist()
        new_data['tls_sni'] = new_data['tls_sni'].fillna('unknown').astype(str).apply(
            lambda x: tls_sni_encoder.transform([x])[0] if x in tls_sni_classes else 0)

        # Mark safe traffic based on domain presence in http_host, tls_sni, or full_url
        new_data['safe_traffic'] = new_data.apply(
            lambda row: any(domain in str(row.get('http_host', '')).lower() or
                            domain in str(row.get('tls_sni', '')).lower() or
                            domain in str(row.get('full_url', '')).lower()
                            for domain in safe_domains),
            axis=1
        )

        # Prepare data for prediction
        X_test = new_data[features]

        # Ensure all features are numeric dtype (already encoded)
        for col in features:
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

        print("Feature dtypes before prediction:")
        print(X_test.dtypes)
        print("Sample data before prediction:")
        print(X_test.head())

        # Predict and decode labels
        predictions = model.predict(X_test)
        predicted_labels = label_encoder.inverse_transform(predictions)
        new_data['Prediction'] = predicted_labels

        # Override Prediction for safe traffic as BENIGN
        new_data.loc[new_data['safe_traffic'], 'Prediction'] = 'BENIGN'

        # Append results to output CSV
        new_data[['timestamp', 'source_ip_raw', 'destination_ip_raw', 'http_host', 'http_uri', 'tls_sni', 'full_url'] +
                 features + ['Prediction']].to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

        # Insert results into DB
        insert_into_db(new_data)

    except Exception as e:
        print("⚠ Error in processing:", e)

if __name__ == '__main__':
    print("🔄 Real-time prediction started...")
    while True:
        process_new_data()
        time.sleep(1)
