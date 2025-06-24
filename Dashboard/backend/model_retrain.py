import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import mysql.connector
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

from config import HOST, USER, PASS, DB, TABLE1

def load_training_data():
    logging.info("Loading original training data from CSV...")
    df = pd.read_csv("training_data.csv")

    # Clean CSV data
    df = df[df['packet_size'] != 'packet_size']
    df['packet_size'] = pd.to_numeric(df['packet_size'], errors='coerce')
    df.dropna(inplace=True)

    required_columns = ['timestamp', 'source_ip', 'destination_ip', 'protocol', 'packet_size',
                        'src_port', 'dst_port', 'tcp_flags', 'ttl', 'Label']

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    df = df[required_columns]

    df['src_port'] = pd.to_numeric(df['src_port'], errors='coerce')
    df['dst_port'] = pd.to_numeric(df['dst_port'], errors='coerce')
    df['ttl'] = pd.to_numeric(df['ttl'], errors='coerce')
    df.dropna(inplace=True)

    return df

def load_db_data():
    logging.info("Loading labeled prediction data from database...")

    conn = mysql.connector.connect(
        host=HOST,
        user=USER,
        password=PASS,
        database=DB
    )
    query = f"""
    SELECT timestamp, source_ip, destination_ip, protocol, packet_size,
           src_port, dst_port, tcp_flags, ttl, prediction AS Label
    FROM {TABLE1}
    WHERE prediction IS NOT NULL AND prediction != ''
    """
    df_db = pd.read_sql(query, conn)
    conn.close()

    # Same cleaning as before
    df_db['packet_size'] = pd.to_numeric(df_db['packet_size'], errors='coerce')
    df_db['src_port'] = pd.to_numeric(df_db['src_port'], errors='coerce')
    df_db['dst_port'] = pd.to_numeric(df_db['dst_port'], errors='coerce')
    df_db['ttl'] = pd.to_numeric(df_db['ttl'], errors='coerce')

    df_db.dropna(inplace=True)
    return df_db

def encode_data(df_train, df_db):
    logging.info("Fitting encoders on combined dataset...")
    # Combine source and destination IPs for consistent encoding
    all_ips = pd.concat([df_train['source_ip'], df_train['destination_ip'],
                         df_db['source_ip'], df_db['destination_ip']]).unique()

    ip_encoder = LabelEncoder()
    ip_encoder.fit(all_ips)

    # Protocol encoder
    all_protocols = pd.concat([df_train['protocol'], df_db['protocol']]).unique()
    protocol_encoder = LabelEncoder()
    protocol_encoder.fit(all_protocols)

    # TCP flags encoder
    all_flags = pd.concat([df_train['tcp_flags'], df_db['tcp_flags']]).unique()
    flags_encoder = LabelEncoder()
    flags_encoder.fit(all_flags)

    # Label encoder (merge labels)
    all_labels = pd.concat([df_train['Label'], df_db['Label']]).unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # Apply encoding to training CSV
    for df in [df_train, df_db]:
        df['source_ip'] = ip_encoder.transform(df['source_ip'])
        df['destination_ip'] = ip_encoder.transform(df['destination_ip'])
        df['protocol'] = protocol_encoder.transform(df['protocol'])
        df['tcp_flags'] = flags_encoder.transform(df['tcp_flags'])
        df['Label'] = label_encoder.transform(df['Label'])

    return df_train, df_db, ip_encoder, protocol_encoder, flags_encoder, label_encoder

def retrain_model():
    logging.info("Starting model retraining...")

    # Load datasets
    df_train = load_training_data()
    df_db = load_db_data()

    # Encode all data
    df_train, df_db, ip_enc, prot_enc, flags_enc, label_enc = encode_data(df_train, df_db)

    # Combine datasets
    combined_df = pd.concat([df_train, df_db], ignore_index=True)

    X = combined_df.drop(['Label', 'timestamp'], axis=1)
    y = combined_df['Label']

    # Compute class weights
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))

    # Build and train model
    model = XGBClassifier(
        learning_rate=1,
        n_estimators=300,
        max_depth=120,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        use_label_encoder=True,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Save model and encoders
    joblib.dump(model, 'attack_detector_model.pkl')
    joblib.dump(ip_enc, 'ip_encoder.pkl')
    joblib.dump(prot_enc, 'protocol_encoder.pkl')
    joblib.dump(flags_enc, 'flags_encoder.pkl')
    joblib.dump(label_enc, 'label_encoder.pkl')

    logging.info("Model retrained and saved successfully.")

def manual_retrain():
    logging.info("Manual retrain started.")
    retrain_model()
    logging.info("Manual retrain finished.")
