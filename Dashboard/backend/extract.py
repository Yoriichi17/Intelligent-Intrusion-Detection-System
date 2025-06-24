import subprocess
import os
import csv
import json
import time

CSV_FILE = "capture_data.csv"
JSON_FILE = "capture_data.json"
TEMP_CSV = "temp_capture.csv"

FIELDS = [
    "frame.time_epoch", "ip.src", "ip.dst", "ip.proto", "frame.len",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", "tcp.flags", "ip.ttl",
    "http.host", "http.request.uri", "ssl.handshake.extensions_server_name"
]

capture_active = True


def get_wifi_interface():
    try:
        output = subprocess.check_output(["tshark", "-D"], universal_newlines=True)
        for line in output.splitlines():
            if "Wi-Fi" in line or "WLAN" in line:
                return line.split(".")[0].strip()
        return output.splitlines()[0].split(".")[0].strip() if output else None
    except subprocess.CalledProcessError:
        return None


def initialize_files():
    headers = [
        "timestamp", "source_ip", "destination_ip", "protocol", "packet_size",
        "src_port", "dst_port", "tcp_flags", "ttl",
        "http_host", "http_uri", "tls_sni", "full_url"
    ]
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, "w", encoding='utf-8') as file:
            json.dump([], file, indent=4)


def capture_packets(interface, duration=1):
    if not interface:
        return

    try:
        command = [
            "tshark",
            "-i", interface,
            "-a", f"duration:{duration}",
            "-f", "ip",
            "-T", "fields"
        ]
        for field in FIELDS:
            command.extend(["-e", field])
        command.extend(["-E", "header=n", "-E", "separator=,", "-E", "quote=n"])

        with open(TEMP_CSV, "w", encoding="utf-8") as file:
            subprocess.run(command, stdout=file, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("Tshark error:", str(e))


# def process_data():
#     new_data = []
#     if not os.path.exists(TEMP_CSV):
#         return

#     with open(TEMP_CSV, "r", encoding='utf-8') as file:
#         reader = csv.reader(file)
#         rows = list(reader)

#     if not rows:
#         return

#     with open(CSV_FILE, "a", newline="", encoding='utf-8') as file:
#         writer = csv.writer(file)
#         for row in rows:
#             row = row + [''] * (len(FIELDS) - len(row))

#             src_port = row[5] if row[5] else row[7]
#             dst_port = row[6] if row[6] else row[8]

#             full_url = ""
#             if row[11] and row[12]:
#                 full_url = f"http://{row[11]}{row[12]}"
#             elif row[13]:
#                 full_url = f"https://{row[13]}"

#             clean_row = [
#                 row[0], row[1], row[2], row[3], row[4],
#                 src_port, dst_port, row[9], row[10],
#                 row[11], row[12], row[13], full_url
#             ]

#             writer.writerow(clean_row)

#             new_data.append({
#                 "timestamp": row[0],
#                 "source_ip": row[1],
#                 "destination_ip": row[2],
#                 "protocol": row[3],
#                 "packet_size": row[4],
#                 "src_port": src_port,
#                 "dst_port": dst_port,
#                 "tcp_flags": row[9],
#                 "ttl": row[10],
#                 "http_host": row[11],
#                 "http_uri": row[12],
#                 "tls_sni": row[13],
#                 "full_url": full_url
#             })

#     if new_data:
#         with open(JSON_FILE, "r", encoding='utf-8') as file:
#             existing_data = json.load(file)

#         existing_data.extend(new_data)

#         with open(JSON_FILE, "w", encoding='utf-8') as file:
#             json.dump(existing_data, file, indent=4)
def process_data():
    new_data = []
    if not os.path.exists(TEMP_CSV):
        return

    with open(TEMP_CSV, "r", encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if not rows:
        return

    with open(CSV_FILE, "a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in rows:
            row = row + [''] * (len(FIELDS) - len(row))

            src_port = row[5] if row[5] else row[7]
            dst_port = row[6] if row[6] else row[8]

            full_url = ""
            if row[11] and row[12]:
                full_url = f"http://{row[11]}{row[12]}"
            elif row[13]:
                full_url = f"https://{row[13]}"

            clean_row = [
                row[0], row[1], row[2], row[3], row[4],
                src_port, dst_port, row[9], row[10],
                row[11], row[12], row[13], full_url
            ]

            writer.writerow(clean_row)

            new_data.append({
                "timestamp": row[0],
                "source_ip": row[1],
                "destination_ip": row[2],
                "protocol": row[3],
                "packet_size": row[4],
                "src_port": src_port,
                "dst_port": dst_port,
                "tcp_flags": row[9],
                "ttl": row[10],
                "http_host": row[11],
                "http_uri": row[12],
                "tls_sni": row[13],
                "full_url": full_url
            })

    # 🛡️ Fix: Handle corrupted or empty JSON file gracefully
    existing_data = []
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding='utf-8') as file:
                existing_data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"[Warning] JSON file is corrupted. Resetting. Error: {e}")
            # Optionally backup the corrupted file
            os.rename(JSON_FILE, JSON_FILE + ".corrupted")
            existing_data = []

    existing_data.extend(new_data)

    with open(JSON_FILE, "w", encoding='utf-8') as file:
        json.dump(existing_data, file, indent=4)


def background_capture():
    initialize_files()
    interface = get_wifi_interface()
    if not interface:
        print("No Wi-Fi interface found.")
        return

    print(f"Using interface: {interface}")

    while capture_active:
        capture_packets(interface=interface, duration=5)
        process_data()
        time.sleep(5)