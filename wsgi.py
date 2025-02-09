import os
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render จะตั้งค่าตัวแปร PORT ให้อยู่แล้ว
    app.run(host="0.0.0.0", port=port)
