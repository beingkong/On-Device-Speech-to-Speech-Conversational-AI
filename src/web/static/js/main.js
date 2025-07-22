document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const statusDiv = document.getElementById('status');
    const conversationDiv = document.getElementById('conversation');

    let ws;
    let audioContext;
    let scriptProcessorNode;
    let mediaStreamSource;
    let isConversationActive = false;

    // Audio playback queue
    let audioQueue = [];
    let isPlaying = false;
    let nextStartTime = 0;

    function setupWebSocket() {
        ws = new WebSocket(`ws://${window.location.host}/audio_stream`);
        ws.binaryType = 'arraybuffer';
        ws.onopen = () => console.log('WebSocket connection established');
        ws.onclose = () => console.log('WebSocket connection closed');
        ws.onerror = (error) => console.error('WebSocket error:', error);
        ws.onmessage = (event) => {
            if (typeof event.data === 'string') {
                const msg = JSON.parse(event.data);
                if (msg.status) statusDiv.textContent = msg.status;
                if (msg.transcription) updateConversation('You', msg.transcription);
            } else {
                if (event.data.byteLength > 0) {
                    const audioChunk = new Float32Array(event.data);
                    audioQueue.push(audioChunk);
                    if (!isPlaying) playAudioQueue();
                }
            }
        };
    }

    function initAudioContext() {
        if (!audioContext || audioContext.state === 'closed') {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            nextStartTime = audioContext.currentTime;
        }
    }

    function playAudioQueue() {
        if (audioQueue.length === 0) { isPlaying = false; return; }
        isPlaying = true;
        initAudioContext();
        const audioChunk = audioQueue.shift();
        const buffer = audioContext.createBuffer(1, audioChunk.length, audioContext.sampleRate);
        buffer.getChannelData(0).set(audioChunk);
        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(audioContext.destination);
        source.start(nextStartTime);
        nextStartTime += buffer.duration;
        source.onended = playAudioQueue;
    }

    function stopPlayback() {
        if (isPlaying) {
            audioQueue = [];
            if (audioContext && audioContext.state === 'running') {
                audioContext.close().then(() => console.log("AudioContext closed for interruption."));
            }
            isPlaying = false;
        }
    }

    async function startConversation() {
        if (isConversationActive) return;
        isConversationActive = true;
        startButton.disabled = true;
        stopButton.disabled = false;
        statusDiv.textContent = 'Connecting...';

        setupWebSocket();
        await new Promise(resolve => {
            if (ws.readyState === WebSocket.OPEN) resolve();
            else ws.onopen = resolve;
        });

        initAudioContext();
        if (audioContext.state === 'suspended') await audioContext.resume();

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 2 } }); // Request stereo to be safe
            
            mediaStreamSource = audioContext.createMediaStreamSource(stream);
            
            // Create a ScriptProcessorNode for robust audio processing
            const bufferSize = 4096;
            scriptProcessorNode = audioContext.createScriptProcessor(bufferSize, 2, 1); // 2 input channels, 1 output channel

            scriptProcessorNode.onaudioprocess = (event) => {
                if (!isConversationActive || ws.readyState !== WebSocket.OPEN) return;

                const inputBuffer = event.inputBuffer;
                const leftChannel = inputBuffer.getChannelData(0);
                const rightChannel = inputBuffer.getChannelData(1);
                
                // Mix stereo to mono
                const monoPcm = new Float32Array(bufferSize);
                for (let i = 0; i < bufferSize; i++) {
                    monoPcm[i] = (leftChannel[i] + rightChannel[i]) / 2;
                }

                // Convert to Int16
                const int16Data = new Int16Array(bufferSize);
                for (let i = 0; i < bufferSize; i++) {
                    int16Data[i] = Math.max(-1, Math.min(1, monoPcm[i])) * 32767;
                }
                
                ws.send(int16Data.buffer);
            };

            mediaStreamSource.connect(scriptProcessorNode);
            scriptProcessorNode.connect(audioContext.destination); // Must be connected to destination to process
            
            statusDiv.textContent = 'Listening...';

        } catch (error) {
            console.error('Error starting conversation:', error);
            statusDiv.textContent = 'Error: Could not start audio stream.';
            stopConversation();
        }
    }

    function stopConversation() {
        if (!isConversationActive) return;
        isConversationActive = false;
        startButton.disabled = false;
        stopButton.disabled = true;
        statusDiv.textContent = 'Conversation ended.';

        if (ws) ws.close();
        if (mediaStreamSource) mediaStreamSource.disconnect();
        if (scriptProcessorNode) scriptProcessorNode.disconnect();
        if (audioContext) audioContext.close();
    }

    function updateConversation(role, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role.toLowerCase()}-message`;
        messageDiv.innerHTML = `<span class="label">${role}:</span> ${text}`;
        conversationDiv.appendChild(messageDiv);
        conversationDiv.scrollTop = conversationDiv.scrollHeight;
    }

    startButton.addEventListener('click', startConversation);
    stopButton.addEventListener('click', stopConversation);
});