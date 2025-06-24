import pandas as pd
import joblib
import sys

# For proper unicode printing, esp. on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model and encoders
model = joblib.load('Models/attack_detector_model.pkl')
label_encoder = joblib.load('Models/label_encoder.pkl')
ip_encoder = joblib.load('Models/ip_encoder.pkl')
flags_encoder = joblib.load('Models/flags_encoder.pkl')

# Load test data
test_data = pd.read_csv('data/test_data.csv')

# Drop rows with missing critical values — if any
essential_cols = ['timestamp', 'source_ip', 'destination_ip', 'protocol', 'packet_size', 'src_port', 'dst_port', 'ttl']
test_data = test_data.dropna(subset=essential_cols)

# ✅ Preserve raw data before modifying
test_data['source_ip_raw'] = test_data['source_ip']
test_data['destination_ip_raw'] = test_data['destination_ip']
test_data['tcp_flags_raw'] = test_data['tcp_flags'] if 'tcp_flags' in test_data.columns else '0'

# Helper function to safely convert mixed values to int
def safe_int(col):
    return test_data[col].astype(str).str.split('#').str[0].str.strip().replace('', '0').astype(float).fillna(0).astype(int)

# Apply conversions
test_data['packet_size'] = safe_int('packet_size')
test_data['src_port'] = safe_int('src_port')
test_data['dst_port'] = safe_int('dst_port')
test_data['ttl'] = safe_int('ttl')
test_data['protocol'] = safe_int('protocol')

# Encode IPs with fallback to 0
test_data['source_ip'] = test_data['source_ip'].apply(
    lambda ip: ip_encoder.transform([ip])[0] if ip in ip_encoder.classes_ else 0)
test_data['destination_ip'] = test_data['destination_ip'].apply(
    lambda ip: ip_encoder.transform([ip])[0] if ip in ip_encoder.classes_ else 0)

# Encode TCP flags with fallback to 0
if 'tcp_flags' in test_data.columns:
    test_data['tcp_flags'] = test_data['tcp_flags'].fillna('0').astype(str).apply(
        lambda flag: flags_encoder.transform([flag])[0] if flag in flags_encoder.classes_ else 0)
else:
    test_data['tcp_flags'] = 0

# Final features for prediction
features_for_model = ['source_ip', 'destination_ip', 'protocol', 'packet_size',
                      'src_port', 'dst_port', 'tcp_flags', 'ttl']

# Mark safe traffic using known domains
safe_domains = ['youtube.com', 'instagram.com', 'facebook.com', 'x.com', 'twitter.com']
test_data['safe_traffic'] = test_data.apply(
    lambda row: any(domain in str(row.get('http_host', '')).lower() or domain in str(row.get('tls_sni', '')).lower()
                    for domain in safe_domains),
    axis=1
)

# Prediction
if test_data.empty:
    print("No valid data to predict. Skipped all rows due to missing or invalid values.")
else:
    X_test = test_data[features_for_model]
    predictions = model.predict(X_test)
    predicted_labels = label_encoder.inverse_transform(predictions)
    test_data['Prediction'] = predicted_labels

    # Override with BENIGN for safe domains
    test_data.loc[test_data['safe_traffic'], 'Prediction'] = 'BENIGN'

    # ✅ Restore original IPs and flags for readable output
    test_data['source_ip'] = test_data['source_ip_raw']
    test_data['destination_ip'] = test_data['destination_ip_raw']
    test_data['tcp_flags'] = test_data['tcp_flags_raw']

    # ✅ Final columns to display (original columns + Prediction)
    display_columns = ['timestamp', 'source_ip', 'destination_ip', 'protocol',
                       'packet_size', 'src_port', 'dst_port', 'tcp_flags', 'ttl',
                       'http_host', 'http_uri', 'tls_sni', 'Prediction']
    
    # 📢 Set pandas display options to NOT truncate columns or use sci notation
    pd.set_option('display.max_columns', None)       # Show all columns
    pd.set_option('display.width', None)             # Don't break columns into new lines
    pd.set_option('display.max_colwidth', None)      # Show full content in each cell
    pd.set_option('display.max_rows', None)          # Show all rows if needed
    pd.set_option('display.float_format', '{:.5f}'.format) # Disable sci notation

    print(test_data[display_columns])

    # ✅ Summary stats
    print("\n✅ Total predictions:", len(test_data))
    print("✅ Prediction breakdown:")
    print(test_data['Prediction'].value_counts())
