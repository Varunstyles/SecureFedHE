import os
import json
import csv

scale_dir = r"website/scale"
csv_files = [
    "scale_clients5.csv", "scale_clients20.csv", "scale_clients50.csv",
    "alpha01.csv", "alpha03.csv", "alpha10.csv",
    "latency_10ms.csv", "latency_50ms.csv", "latency_100ms.csv",
    "arch_simplecnn.csv", "arch_resnet18.csv",
    "dlg_results.csv", "poisoning_results.csv", "mia_results.csv"
]

all_data = {}

for filename in csv_files:
    filepath = os.path.join(scale_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Some files might have spaces in headers, so we strip them
            rows = []
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items()}
                rows.append(clean_row)
            all_data[f"scale/{filename}"] = rows

# Now rewrite loadCSV in app.js
app_js_path = r"website/app.js"
with open(app_js_path, "r", encoding="utf-8") as f:
    app_js = f.read()

# Create the JS variable
js_data = f"const SCALE_DATA = {json.dumps(all_data, separators=(',', ':'))};\n"

# Find loadCSV and replace it
import re

old_load_csv = r"""function loadCSV\(path, callback\) \{
    const promise = fetch\(path\)
        \.then\(r => \{
            if \(!r\.ok\) throw new Error\(`Failed to load \$\{path\}`\);
            return r\.text\(\);
        \}\)
        \.then\(text => \{
            const rows = text\.trim\(\)\.split\('\\n'\);
            if \(rows\.length < 2\) return \[\];
            const headers = rows\[0\]\.split\(','\)\.map\(h => h\.trim\(\)\);
            return rows\.slice\(1\)\.map\(line => \{
                const vals = line\.split\(','\);
                const obj = \{\};
                headers\.forEach\(\(h, i\) => \{ obj\[h\] = \(vals\[i\] \|\| ''\)\.trim\(\); \}\);
                return obj;
            \}\);
        \}\);
    if \(callback\) promise\.then\(callback\);
    return promise;
\}"""

new_load_csv = """function loadCSV(path, callback) {
    const promise = Promise.resolve(SCALE_DATA[path] || []);
    if (callback) promise.then(callback);
    return promise;
}"""

app_js = re.sub(old_load_csv, new_load_csv, app_js)

# Insert SCALE_DATA at the very top
if "const SCALE_DATA" not in app_js:
    app_js = js_data + app_js

with open(app_js_path, "w", encoding="utf-8") as f:
    f.write(app_js)

print("Embedded CSV data successfully.")
