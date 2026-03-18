import subprocess
import sys
import os

# Activate virtualenv
INTERP = os.path.join(os.getcwd(), "venv", "bin", "python")
if sys.executable != INTERP:
    os.execl(INTERP, INTERP, *sys.argv)

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Port provided by Passenger
port = os.environ.get("PORT", "8501")

# Launch Streamlit
subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--server.port", port,
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false",
    "--server.fileWatcherType", "none",
])

# Minimal WSGI app required by Passenger
def application(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"Streamlit app is running."]