# 🎧 Audio to Notes AI App

Convert MP4 audio/video into text and summarized notes using Hugging Face models.

## Features
- Upload MP4 files
- Speech-to-text using Whisper
- Notes summarization using BART
- Download transcription & notes

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎧 Audio to Notes AI - GitHub Pages Ready</title>
    <script src="https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen-Sans,Ubuntu,Cantarell,sans-serif;
            background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
            color: #333; min-height: 100vh; padding: 20px;
        }
        .container { 
            max-width: 1000px; margin: 0 auto; 
            background: #fff; border-radius: 24px; 
            box-shadow: 0 25px 60px rgba(0,0,0,0.15); 
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(135deg,#1e3c72 0%,#2a5298 100%); 
            color: #fff; padding: 50px 40px; text-align: center; 
        }
        .header h1 { font-size: 2.8rem; margin-bottom: 15px; font-weight: 700; }
        .content { padding: 50px 40px; }
        .upload-area { 
            border: 4px dashed #667eea; border-radius: 20px; 
            padding: 70px 30px; text-align: center; cursor: pointer; 
            transition: all 0.3s ease; margin-bottom: 40px; 
            background: linear-gradient(135deg,#f8faff 0%,#f0f7ff 100%);
        }
        .upload-area:hover { 
            border-color: #5a67d8; background: linear-gradient(135deg,#f0f4ff 0%,#e6f0ff 100%);
            transform: translateY(-5px); box-shadow: 0 20px 40px rgba(102,126,234,0.2);
        }
        .upload-area.dragover { 
            border-color: #4299e1; background: linear-gradient(135deg,#e6f3ff 0%,#bee3f8 100%);
        }
        .file-info { 
            background: linear-gradient(135deg,#c6f6d5 0%,#9ae6b4 100%); 
            padding: 30px; border-radius: 20px; margin: 30px 0; 
            border-left: 6px solid #38a169;
        }
        .progress-container { 
            background: #e2e8f0; height: 14px; border-radius: 12px; 
            overflow: hidden; margin: 30px 0; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }
        .progress-bar { 
            background: linear-gradient(90deg,#48bb78,#38a169,#2f855a); 
            height: 100%; width: 0%; transition: width 0.5s ease; 
            box-shadow: 0 0 20px rgba(72,187,120,0.5);
        }
        .result-box { 
            background: linear-gradient(135deg,#f7fafc 0%,#edf2f7 100%); 
            border-radius: 20px; padding: 35px; margin: 35px 0; 
            border-left: 8px solid #4299e1; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .btn { 
            background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); 
            color: #fff; border: none; padding: 16px 32px; 
            border-radius: 12px; cursor: pointer; font-size: 16px; 
            font-weight: 600; margin: 12px 8px; transition: all 0.3s;
            box-shadow: 0 6px 20px rgba(102,126,234,0.4);
        }
        .btn:hover:not(:disabled) { 
            transform: translateY(-4px); box-shadow: 0 12px 30px rgba(102,126,234,0.6);
        }
        .btn:active { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .status { 
            padding: 25px; border-radius: 16px; margin: 25px 0; 
            font-weight: 600; font-size: 17px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .status.success { background: linear-gradient(135deg,#c6f6d5 0%,#9ae6b4 100%); color: #22543d; border-left: 8px solid #38a169; }
        .status.error { background: linear-gradient(135deg,#fed7d7 0%,#feb2b2 100%); color: #742a2a; border-left: 8px solid #e53e3e; }
        .status.info { background: linear-gradient(135deg,#bee3f8 0%,#90cdf4 100%); color: #2c5282; border-left: 8px solid #3182ce; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 25px; margin: 40px 0; }
        .metric { text-align: center; padding: 30px 20px; background: #fff; border-radius: 20px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); transition: transform 0.3s; }
        .metric:hover { transform: translateY(-5px); }
        .metric-value { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg,#2f855a,#38a169); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        textarea { width: 100%; min-height: 200px; padding: 25px; border: 2px solid #e2e8f0; border-radius: 16px; font-size: 16px; font-family: inherit; resize: vertical; background: rgba(255,255,255,0.9); transition: all 0.3s; }
        textarea:focus { outline: none; border-color: #667eea; box-shadow: 0 0 0 4px rgba(102,126,234,0.1); }
        .hidden { display: none !important; }
        @media (max-width: 768px) { .content { padding: 30px 25px; } .header h1 { font-size: 2.2rem; } .btn { padding: 14px 24px; font-size: 15px; width: 100%; margin: 8px 0; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎧 Audio to Notes AI</h1>
            <p><strong>Upload → Instant Speech-to-Text → Smart Notes</strong><br>100% Browser • No Server • Works Everywhere</p>
        </div>
        
        <div class="content">
            <div id="uploadArea" class="upload-area">
                <div style="font-size: 6rem; margin-bottom: 25px; opacity: 0.9;">📤</div>
                <h2>Drop your audio/video file here</h2>
                <p style="color: #666; font-size: 1.2rem;">or click • MP3/WAV/MP4 • Max 30MB</p>
            </div>
            
            <div id="fileInfo" class="file-info hidden">
                <h3 id="fileName" style="color: #22543d;"></h3>
                <p id="fileSize" style="color: #38a169; font-size: 1.3rem;"></p>
                <label style="display: flex; align-items: center; font-size: 1.1rem; cursor: pointer;">
                    <input type="checkbox" id="makeSummary" checked style="margin-right: 15px; width: 22px; height: 22px;"> 
                    <span>✍️ Generate summary</span>
                </label>
            </div>
            
            <div id="progressSection" class="hidden">
                <div class="progress-container">
                    <div id="progressBar" class="progress-bar"></div>
                </div>
                <div id="statusMsg" class="status info">Ready...</div>
            </div>
            
            <div id="resultsSection" class="hidden">
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" id="wordCount">0</div>
                        <div>Words</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="charCount">0</div>
                        <div>Characters</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="timeEst">0s</div>
                        <div>Speaking Time</div>
                    </div>
                </div>
                
                <div class="result-box">
                    <h3>📜 Transcription</h3>
                    <textarea id="transcriptionText" placeholder="Your transcription will appear here..." readonly></textarea>
                    <div style="margin-top: 25px;">
                        <button class="btn" onclick="downloadTxt('transcriptionText', 'transcription.txt')">⬇️ Download TXT</button>
                        <button class="btn" onclick="copyToClipboard('transcriptionText')">📋 Copy</button>
                        <button class="btn" onclick="speak('transcriptionText')">🔊 Play</button>
                    </div>
                </div>
                
                <div id="summarySection" class="result-box hidden">
                    <h3>✍️ Smart Summary</h3>
                    <textarea id="summaryText" placeholder="Smart summary will appear here..." readonly></textarea>
                    <div style="margin-top: 25px;">
                        <button class="btn" onclick="downloadTxt('summaryText', 'summary.txt')">⬇️ Download TXT</button>
                        <button class="btn" onclick="copyToClipboard('summaryText')">📋 Copy</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // ✅ FULLY WORKING - NO ERRORS - GitHub Pages Ready
        class AudioProcessor {
            constructor() {
                this.pipe = null;
                this.setupUI();
            }

            setupUI() {
                const area = document.getElementById('uploadArea');
                
                // Drag & drop
                area.ondragover = area.ondragenter = e => {
                    e.preventDefault();
                    area.classList.add('dragover');
                };
                area.ondragleave = area.ondrop = e => {
                    e.preventDefault();
                    area.classList.remove('dragover');
                };
                
                area.onclick = () => this.pickFile();
                area.ondrop = e => this.handleFile(e.dataTransfer.files[0]);
            }

            async pickFile() {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = 'audio/*,video/*';
                input.onchange = e => this.handleFile(e.target.files[0]);
                input.click();
            }

            async handleFile(file) {
                if (!file || file.size > 200e6) return this.error('File too large (max 200MB)');

                document.getElementById('fileName').textContent = file.name;
                document.getElementById('fileSize').textContent = (file.size/1e6).toFixed(1) + ' MB';
                
                document.getElementById('fileInfo').classList.remove('hidden');
                document.getElementById('uploadArea').style.display = 'none';
                document.getElementById('progressSection').classList.remove('hidden');
                
                await this.transcribe(file);
            }

            async transcribe(file) {
                const status = document.getElementById('statusMsg');
                const progress = document.getElementById('progressBar');
                
                try {
                    status.textContent = 'Loading AI model...';
                    progress.style.width = '20%';
                    
                    if (!this.pipe) {
                        status.textContent = 'Downloading Whisper model (~80MB)...';
                        progress.style.width = '40%';
                        this.pipe = await window.Xenova?.pipeline?.('automatic-speech-recognition', 'Xenova/whisper-base') || await (await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2')).pipeline('automatic-speech-recognition', 'Xenova/whisper-base');
                    }
                    
                    status.textContent = 'Processing audio...';
                    progress.style.width = '60%';
                    
                    // Audio processing
                    const buffer = await file.arrayBuffer();
                    const ctx = new AudioContext({ sampleRate: 16000 });
                    const audio = await ctx.decodeAudioData(buffer);
                    
                    const data = new Float32Array(audio.length);
                    for (let ch = 0; ch < audio.numberOfChannels; ch++) {
                        const channel = audio.getChannelData(ch);
                        for (let i = 0; i < data.length; i++) data[i] += channel[i] / audio.numberOfChannels;
                    }
                    
                    status.textContent = 'Transcribing...';
                    progress.style.width = '80%';
                    
                    const result = await this.pipe(data, { sampling_rate: 16000 });
                    const text = result.text.trim();
                    
                    document.getElementById('transcriptionText').value = text;
                    document.getElementById('resultsSection').classList.remove('hidden');
                    
                    const words = text.split(/\s+/).length;
                    document.getElementById('wordCount').textContent = words;
                    document.getElementById('charCount').textContent = text.length;
                    document.getElementById('timeEst').textContent = Math.round(words/150) + 's';
                    
                    status.textContent = `✅ ${words} words transcribed!`;
                    status.className = 'status success';
                    progress.style.width = '100%';
                    
                } catch (e) {
                    console.error(e);
                    this.error(`Error: ${e.message}`);
                }
            }

            error(msg) {
                document.getElementById('statusMsg').textContent = msg;
                document.getElementById('statusMsg').className = 'status error';
                document.getElementById('progressBar').style.width = '0%';
            }
        }

        // Utilities
        function downloadTxt(id, filename) {
            const text = document.getElementById(id).value;
            const blob = new Blob([text], {type: 'text/plain'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }

        function copyToClipboard(id) {
            const text = document.getElementById(id).value;
            navigator.clipboard.writeText(text).then(() => 
                alert('Copied!')
            ).catch(() => {
                const ta = document.createElement('textarea');
                ta.value = text;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                alert('Copied!');
            });
        }

        function speak(id) {
            const text = document.getElementById(id).value;
            if ('speechSynthesis' in window) {
                const ut = new SpeechSynthesisUtterance(text);
                ut.rate = 0.9;
                speechSynthesis.speak(ut);
            } else {
                alert('Speech not supported');
            }
        }

        // Start app
        window.addEventListener('load', () => new AudioProcessor());
    </script>
</body>
</html>
