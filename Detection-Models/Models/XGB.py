import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Detection-Models/data/training_data.csv")
df = df[df['packet_size'] != 'packet_size']
df['packet_size'] = pd.to_numeric(df['packet_size'], errors='coerce')
df.dropna(inplace=True)

required_columns = ['timestamp', 'source_ip', 'destination_ip', 'protocol', 'packet_size',
                    'src_port', 'dst_port', 'tcp_flags', 'ttl', 'http_host', 'http_uri', 'tls_sni', 'Label']

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in dataset: {missing_cols}")

df = df[required_columns]

df['src_port'] = pd.to_numeric(df['src_port'], errors='coerce')
df['dst_port'] = pd.to_numeric(df['dst_port'], errors='coerce')
df['ttl'] = pd.to_numeric(df['ttl'], errors='coerce')
df.dropna(inplace=True)

ip_encoder = LabelEncoder()
df['source_ip'] = ip_encoder.fit_transform(df['source_ip'])
df['destination_ip'] = ip_encoder.fit_transform(df['destination_ip'])

protocol_encoder = LabelEncoder()
df['protocol'] = protocol_encoder.fit_transform(df['protocol'])

flags_encoder = LabelEncoder()
df['tcp_flags'] = flags_encoder.fit_transform(df['tcp_flags'])

label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

for col in ['http_host', 'http_uri', 'tls_sni']:
    df[col] = df[col].fillna('unknown')
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    joblib.dump(enc, f'Detection-Models/Models/{col}_encoder.pkl')

joblib.dump(ip_encoder, 'Detection-Models/Models/ip_encoder.pkl')
joblib.dump(protocol_encoder, 'Detection-Models/Models/protocol_encoder.pkl')
joblib.dump(flags_encoder, 'Detection-Models/Models/flags_encoder.pkl')
joblib.dump(label_encoder, 'Detection-Models/Models/label_encoder.pkl')

X = df.drop(['Label', 'timestamp'], axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Apply SMOTE to oversample rare classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Recompute class weights on balanced data
classes = np.unique(y_train_res)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_res)
class_weight_dict = dict(zip(classes, weights))

# ✅ Build improved XGBoost model with better hyperparameters
model = XGBClassifier(
    learning_rate=0.1,
    n_estimators=500,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.3,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

plot_importance(model, max_num_features=10)
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

joblib.dump(model, 'Detection-Models/Models/attack_detector_model.pkl')
print("Model and encoders saved successfully.")