import modal

app = modal.App("audio-cnn")

# Modal image definition for remote training environment.
# - Based on minimal Debian image
# - Installs Python and system dependencies
# - Downloads and extracts ESC-50 dataset
# - Copies audio data to /opt/esc50-data
# - Adds local model.py source
image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))


# Volume for storing input data (ESC-50 audio files)
volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
# Volume for storing training outputs
modal_volume = modal.Volume.from_name("esc50-data", create_if_missing=True)



@app.function(
    image=image,
    gpu="A10",
    volumes={
        "/data": volume,    # Mounts input data volume at /data
        "/models": modal_volume  # Mounts output volume at /models
    },
    timeout=60 * 60 * 3  # 3 hours (in seconds)
)
def train():
    print("training")

@app.local_entrypoint()
def main():
    train.remote()
