import os
import csv
import json
import time
import sys
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

CSV_FILE = "Detection-Models/data/capture_data.csv"
JSON_FILE = "Detection-Models/data/capture_data.json"
TEMP_CSV = "Detection-Models/data/temp_capture.csv"

def get_wifi_interface():
    try:
        output = subprocess.check_output(["tshark", "-D"], universal_newlines=True)
        for line in output.splitlines():
            if "Wi-Fi" in line or "WLAN" in line:
                return line.split(".")[0].strip()
    except subprocess.CalledProcessError:
        print("❌ Error detecting Wi-Fi interface.")
        return None

def initialize_files():
    headers = [
        "timestamp", "source_ip", "destination_ip", "protocol", "packet_size",
        "src_port", "dst_port", "tcp_flags", "ttl", "http_host", "http_uri", "tls_sni"
    ]
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, "w", encoding='utf-8') as file:
            json.dump([], file, indent=4)

def capture_packets(interface, capture_duration=1):
    if not interface:
        print("⚠ No Wi-Fi interface found.")
        return

    command = f'''
tshark -i {interface} -a duration:{capture_duration} -T fields \
-e frame.time_epoch \
-e ip.src \
-e ip.dst \
-e ip.proto \
-e frame.len \
-e tcp.srcport \
-e tcp.dstport \
-e tcp.flags \
-e ip.ttl \
-e http.host \
-e http.request.uri \
-e ssl.handshake.extensions_server_name \
-E header=n -E separator=, -E quote=n > {TEMP_CSV}
'''
    print(f"📡 Capturing packets for {capture_duration} seconds on interface {interface}...")
    os.system(command)

def process_and_save_data():
    new_data = []

    if not os.path.exists(TEMP_CSV):
        print("⚠ No packets captured.")
        return

    with open(TEMP_CSV, "r", encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if not rows:
        print("⚠ No valid data in capture.")
        return

    with open(CSV_FILE, "a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in rows:
            # Pad missing fields up to 12 columns
            row = row + [''] * (12 - len(row))
            writer.writerow(row)

            full_url = ""
            if row[9] and row[10]:
                full_url = f"http://{row[9]}{row[10]}"
            elif row[11]:
                full_url = f"https://{row[11]}"

            new_data.append({
                "timestamp": row[0],
                "source_ip": row[1],
                "destination_ip": row[2],
                "protocol": row[3],
                "packet_size": row[4],
                "src_port": row[5],
                "dst_port": row[6],
                "tcp_flags": row[7],
                "ttl": row[8],
                "http_host": row[9],
                "http_uri": row[10],
                "tls_sni": row[11],
                "full_url": full_url
            })

    if new_data:
        with open(JSON_FILE, "r", encoding='utf-8') as file:
            existing_data = json.load(file)

        existing_data.extend(new_data)

        with open(JSON_FILE, "w", encoding='utf-8') as file:
            json.dump(existing_data, file, indent=4)

        print(f"✅ Updated {CSV_FILE} and {JSON_FILE} with new data.")
    else:
        print("⚠ No valid packets to save.")

def main(interval=1):
    initialize_files()
    wifi_interface = get_wifi_interface()

    if not wifi_interface:
        print("❌ No Wi-Fi interface detected. Exiting.")
        return

    while True:
        capture_packets(interface=wifi_interface, capture_duration=interval)
        process_and_save_data()
        time.sleep(interval)

if __name__ == "__main__":
    main(interval=1)
