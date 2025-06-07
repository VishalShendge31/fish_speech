# fish_speech
Fish speech 


git clone https://github.com/fishaudio/fish-speech.git

cd fish-speech
git checkout tags/v1.5.0 --force

python3 -m venv fish_speech
source fish_speech/bin/activate

sudo apt update
sudo add-apt-repository universe
sudo apt update

# Install Packages
sudo apt install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
pip install pyaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install huggingface_hub
pip install triton
pip install .

huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5

#Inference
## Generate semantic tokens from text

python3 tools/vqgan/inference.py \
  -i /workspaces/fish_speech/fish-speech/LJSpeech-1.1/wavs/LJ001-0001.wav \
  -o /workspaces/fish_speech/fish-speech/LJSpeech-1.1/wavs/LJ001-0001.npy \
  --checkpoint-path /workspaces/fish_speech/fish-speech/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth

  
python3 tools/llama/generate.py --text "Hello everyone! I am Vishal. Just testing how it works" --prompt-text "prompt_wav_text" --prompt-tokens vishal.npy --checkpoint-path /workspaces/fish_speech/fish-speech/checkpoints/fish-speech-1.5 --num-samples 2 --device {device}'