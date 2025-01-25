# Speech-to-Speech Chatbot

A real-time speech-to-speech chatbot powered by **Whisper Small**, **Llama 3.2**, and **Kokoro-82M**. Designed for natural conversations, this chatbot offers continuous speech recognition, natural language understanding, and real-time voice responses. It is optimized to work on **Apple Silicon** devices.

---

## 🌟 Features

- **Continuous Speech Recognition**: Powered by Whisper MLX.
- **Natural Language Understanding**: Responses generated using Llama 3.2.
- **Real-Time Text-to-Speech**: Synthesized voice responses with Kokoro-82M.
- **Voice Customization**: Supports multiple voices.
- **Streaming Responses**: Smooth, real-time interactions.

---

## 🛠️ Setup

Before you begin, ensure you have the following dependencies and models installed:

### 1. Download Required Files
Download the following files from [Google Drive](https://drive.google.com/drive/folders/1EysDT7TAcaMEz-C6i3EVykI4jKsHIBWy?usp=sharing) and place them in the specified directories:

| File                      | Directory            |
|---------------------------|----------------------|
| `kokoro-v0_19.onnx`       | Root directory       |
| `weights.npz`             | `mlx_models/small/` |
| `voices.json`             | Root directory       |

### 2. Install Dependencies
Install the necessary Python dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate        # For Mac/Linux
.venv\Scripts\activate           # For Windows
pip install -r requirements.txt
```
### 3. Additional Requirements (for Mac users)
For Mac users, install `espeak-ng` for text-to-speech:
```bash
brew install espeak-ng
export ESPEAK_DATA_PATH=/opt/homebrew/share/espeak-ng-data
```
### 4. Pull Llama 3.2 Model
Download the **Llama 3.2** model using `ollama`:
```bash
ollama pull llama3.2
```

## 🚀 Usage

### Run the Chatbot
Run the chatbot application:
```bash
python main.py
```

The program will start listening for voice input.  
Speak naturally, and the chatbot will respond with synthesized speech.  
Press `Ctrl+C` to stop.

---

## 📂 Directory Structure

```plaintext
speech-to-speech/
├── LICENSE
├── README.md
├── main.py
├── requirements.txt
├── mlx_models/
│   └── small/
│       └── weights.npz
└── voices.json
```

## 📝 Notes

- This project is optimized for **Apple Silicon** but may work on other platforms with appropriate configuration.
- Ensure all required files are downloaded and placed in their respective directories as described above.

---

## 📖 Learn More

For more details about the underlying technologies, visit:
- [Whisper MLX](https://github.com/openai/whisper)
- [Llama](https://ollama.ai)
- [Kokoro-82M](https://github.com/thewh1teagle/kokoro-onnx)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
