import re

app_js_path = r"website/app.js"
with open(app_js_path, "r", encoding="utf-8") as f:
    app_js = f.read()

# Replace 'new Chart(ctx,' with:
# const existingChart = Chart.getChart(ctx.canvas.id); if (existingChart) existingChart.destroy();
# chartInstances[ctx.canvas.id] = new Chart(ctx,

replacement_logic = """
        const chartId = ctx.canvas.id;
        const existingChart = Chart.getChart(chartId);
        if (existingChart) existingChart.destroy();
        new Chart(ctx,"""

app_js = app_js.replace("new Chart(ctx,", replacement_logic)

with open(app_js_path, "w", encoding="utf-8") as f:
    f.write(app_js)
print("Updated chart instantiation logic.")
