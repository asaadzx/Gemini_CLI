# 🤖 Gemini Chatbot - Your AI Companion

<div align="center">

![Gemini Logo](https://img.shields.io/badge/Powered%20by-Google%20Gemini-blue)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

</div>

A powerful, multimodal CLI chatbot leveraging Google's Generative AI technology. Experience seamless conversations with text and images through a beautiful Rich UI. 🚀

## ✨ Key Features

- 🔄 **Real-time Response Streaming**
- 🖼️ **Image & Text Processing**
- 🎨 **Rich, Colorful Interface**
- 🛠️ **Customizable Parameters**
- 📚 **Comprehensive Help System**
- 🤝 **User-Friendly Design**

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- [Google AI Studio API Key](https://aistudio.google.com/)

### Setup in 4 Easy Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/gemini-chatbot.git
cd gemini-cli

# 2. Install dependencies
pip install requirements-detector

# 3. Configure API key
export GOOGLE_API_KEY="your-api-key-here"

# 4. Launch the chatbot
python Main.py
```

### Advanced Configuration
```bash
python Main.py --model gemini-2.0-flash-exp --api-version v1alpha
```

## 💡 Usage Guide

### Available Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `/image <path>` | Process images | `/image cat.jpg` |
| `help` | View commands | `help` |
| `exit/quit/bye` | Close chatbot | `exit` |
| `<text>` | Send message | `Hello!` |

### Configuration Options

```bash
--model (-m)              # Select Gemini model
--api-version (-a)        # Set API version
--voice (-v)             # Enable voice output
--temperature (-t)        # Control creativity
--max-output-tokens      # Set response length
```

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -am 'Add something amazing'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📄 License

This project is MIT licensed - see [LICENSE](LICENSE) for details.

## 🙌 Acknowledgments

- **Google AI** for Gemini API
- **Rich** library for CLI enhancement
- **Community** for support and contributions

<div align="center">

### Made with ❤️ by Asaadzx

</div>
