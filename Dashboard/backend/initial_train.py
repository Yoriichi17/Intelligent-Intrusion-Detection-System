import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("training_data.csv")

# Drop rows where header is repeated or packet_size has garbage values
df = df[df['packet_size'] != 'packet_size']
df['packet_size'] = pd.to_numeric(df['packet_size'], errors='coerce')

# Drop rows with NaNs
df.dropna(inplace=True)

# Required columns
required_columns = ['timestamp', 'source_ip', 'destination_ip', 'protocol', 'packet_size',
                    'src_port', 'dst_port', 'tcp_flags', 'ttl', 'Label']

# Check for missing columns
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in dataset: {missing_cols}")

# Filter and reorder
df = df[required_columns]

# Convert problematic columns to numeric
df['src_port'] = pd.to_numeric(df['src_port'], errors='coerce')
df['dst_port'] = pd.to_numeric(df['dst_port'], errors='coerce')
df['ttl'] = pd.to_numeric(df['ttl'], errors='coerce')

# Drop any remaining NaNs after conversion
df.dropna(inplace=True)

# Encode IPs
ip_encoder = LabelEncoder()
df['source_ip'] = ip_encoder.fit_transform(df['source_ip'])
df['destination_ip'] = ip_encoder.fit_transform(df['destination_ip'])

# Encode protocol
protocol_encoder = LabelEncoder()
df['protocol'] = protocol_encoder.fit_transform(df['protocol'])

# Encode TCP flags
flags_encoder = LabelEncoder()
df['tcp_flags'] = flags_encoder.fit_transform(df['tcp_flags'])

# Encode labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Save encoders
joblib.dump(ip_encoder, 'ip_encoder.pkl')
joblib.dump(protocol_encoder, 'protocol_encoder.pkl')
joblib.dump(flags_encoder, 'flags_encoder.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Features and labels
X = df.drop(['Label', 'timestamp'], axis=1)
y = df['Label']

# Train-test split (no SMOTE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Compute class weights
classes = np.unique(y)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, weights))

# Build improved model
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

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(model, 'attack_detector_model.pkl')
print("Model and encoders saved successfully.")
