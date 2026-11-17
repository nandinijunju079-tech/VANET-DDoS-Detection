
import json
import glob
import os

BASE = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset\Extracted_DOS\Expanded_Test\InTAS_highway_2_dosAttack"

files = glob.glob(os.path.join(BASE, "*.json"))

if not files:
    raise FileNotFoundError("No JSON files found.")

with open(files[0], "r", encoding="utf-8") as f:
    data = json.load(f)

print("===== JSON FILE =====")
print(os.path.basename(files[0]))

print("\n===== TOP-LEVEL TYPE =====")
print(type(data).__name__)

if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))

    if "messages" in data:
        messages = data["messages"]
    else:
        messages = [data]

elif isinstance(data, list):
    messages = data
else:
    raise ValueError("Unexpected JSON structure.")

print("\n===== FIRST MESSAGE =====")

if messages:
    msg = messages[0]
    print(json.dumps(msg, indent=4)[:6000])

    print("\n===== MESSAGE FIELDS =====")
    print(list(msg.keys()))

    print("\n===== RECEIVER DATA =====")
    print(json.dumps(msg.get("receiver", {}), indent=4))

    print("\n===== SENDER DATA =====")
    print(json.dumps(msg.get("sender", {}), indent=4))
else:
    print("No messages found.")
