This is a real-time conversational system for two-way speech communication with AI models, utilizing a continuous streaming architecture for fluid conversations with immediate responses and natural interruption handling. All components of this system are run locally [on CPU, in my test system].

## System Architecture & Technologies

The system employs a multi-threaded architecture, where each component operates independently but is integrated through a queue management system to ensure performance and responsiveness. This design maintains a natural conversational flow, powered by several specialized AI models:

- **Speech Recognition**: Whisper:whisper-tiny.en (OpenAI)
- **Voice Activity Detection**: Pyannote:pyannote/segmentation-3.0
- **Language Model**: LM Studio:llama-3.2-1b-instruct
- **Voice Synthesis**: Kokoro:hexgrad/Kokoro-82M

![System Overview](assets/system_design.png)


The audio processing pipeline begins with continuous monitoring through Voice Activity Detection (VAD) using Pyannote.audio's Segment.30 model, offering precise speech detection with low latency (2-3ms per frame). When speech is detected, the system immediately captures and processes the audio through OpenAI's Whisper-tiny.en model, optimized for English speech recognition (39M parameters). The transcribed text feeds into a locally hosted Llama-2 1B model through LM Studio, which generates responses token by token. Finally, the Web Kokoro 82M model handles voice synthesis, enabling immediate audio generation as soon as the first tokens become available.

## Technical Implementation

The system leverages a carefully selected stack of AI models and technologies:

### Speech Processing Stack
- **Voice Activity Detection**: Pyannote.audio Segment.30
  - Optimized for real-time speech detection
  - Low latency operation (2-3ms per frame)
  - Accurate speaker segmentation

- **Speech Recognition**: Whisper-tiny.en
  - Lightweight model optimized for English
  - 39M parameters for efficient processing
  - Real-time transcription capabilities

### Language Processing
- **Language Model**: LM Studio with Llama-2 1B
  - Local inference for reduced latency
  - Streaming token generation
  - Context-aware responses
  - Optimized for conversational AI

### Voice Synthesis
- **Text-to-Speech**: Web Kokoro 82M
  - Neural voice synthesis
  - Low latency generation
  - Support for voice mixing and modification
  - Efficient tensor operations

The implementation includes sophisticated queue management for efficient data flow, with separate queues handling text processing and audio generation. This multi-threaded architecture ensures that computationally intensive tasks don't impact system responsiveness, while the interrupt handling system enables natural conversation flow through immediate response to user input.

```ascii
                              System Overview
+--------------------------------------------------------------------------------+
|                        Multi-Threaded Architecture                             |
|                                                                                |
|  +-------------+     +--------------+     +-------------+     +-------------+  |
|  |   Audio     |     |   Speech     |     |    LLM      |     |   Voice     |  |
|  | Monitoring  |---->| Recognition  |---->| Processing  |---->| Synthesis   |  |
|  |  Thread     |     |   (Whisper)  |     |   Stream    |     |   Engine    |  |
|  +-------------+     +--------------+     +-------------+     +-------------+  |
|        ↑                                                            |          |
|        +------------------------------------------------------------+          |
|                           Interrupt Feedback Loop                              |
+--------------------------------------------------------------------------------+
```

Audio processing starts with continuous monitoring using Voice Activity Detection (VAD) via pyannote.audio. When speech is detected, the system captures and processes the audio using the Whisper model for transcription. The transcribed text is then fed into a streaming language model, which generates responses token by token. This allows for immediate voice synthesis as soon as the first tokens are available.

## Intelligent Text Processing

The system uses the `TextChunker` component to prepare text for voice generation. This component analyzes incoming text streams from the language model and splits them into chunks suitable for the voice synthesizer. It uses a priority system for semantic breaks and punctuation to determine the best places to split the text, ensuring natural-sounding speech output. The `TextChunker` prioritizes splitting at the end of sentences, but will also split at semantic breaks like "however" or "and" if a sentence is too long. The first sentence is typically shorter, and subsequent sentences are longer. This is done to get the first part of the response out quickly, and then to keep the audio flowing smoothly.

```ascii
                           Text Processing Pipeline
+--------------------------------------------------------------------------------+
|                                                                                |
|  Input Stream   +-------------+    Semantic      +-------------+    Audio      |
|  Tokens     --->| TextChunker |--->Analysis  --->| Generation  |--->Output     |
|                 |             |    & Breaks      | Queue       |               |
|                 +-------------+                  +-------------+               |
|                      ↓                              ↓                          |
|               Priority-based                 Continuous Stream                 |
|               Break Detection               Processing                         |
+--------------------------------------------------------------------------------+
```

The `TextChunker` uses semantic break points with assigned priorities. These range from strong breaks (priority 4) for discourse markers like "however" and "therefore," to basic connectors (priority 2) for words like "and" and "but." Punctuation also has a priority, with periods, question marks, and exclamation points having the highest priority (5). When the `TextChunker` receives a text stream, it buffers the text until it reaches a target length or a high-priority break point. It then splits the text at the best break point and sends the chunk to the audio queue. The remaining text is kept in the buffer for further processing. This approach helps the generated speech maintain a natural rhythm and pace.

## Audio Queue Management

The system uses a queue management architecture to handle both input and output audio streams. The `AudioGenerationQueue` acts as the central coordinator, managing multiple queues that handle different aspects of the audio processing pipeline.

```ascii
                           Queue Management System
+--------------------------------------------------------------------------------+
|                                                                                |
|    Input Pipeline              Processing Pipeline         Output Pipeline     |
|  +--------------+            +-----------------+         +--------------+      |
|  | Speech Input |            | AudioGeneration |         | Playback     |      |
|  | Queue        |----------->| Queue System    |-------->| Stream       |      |
|  +--------------+            +-----------------+         +--------------+      |
|        ↓                            ↓                          ↓               |
|    VAD-based                   Multi-threaded             Interrupt            |
|    Detection                   Processing                 Monitoring           |
+--------------------------------------------------------------------------------+
```

The queue system uses non-blocking operations throughout the pipeline, allowing it to remain responsive under load. If a user starts speaking while the AI is responding, the system can immediately stop the current processing, clear the relevant queues, and switch to processing the new input.

## Voice Generation System

The voice generation component manages speech synthesis using a tensor-based pipeline. This system supports dynamic voice switching and mixing, achieved through a voice embedding processor that can combine multiple voice characteristics in real-time.

```ascii
                           Voice Processing System
+--------------------------------------------------------------------------------+
|                                                                                |
|   Voice Management                    Generation Pipeline                      |
|  +----------------+                  +------------------+                      |
|  | Voice Registry |----------------->| Tensor Pipeline  |                      |
|  | - Voice Files  |                  | - Weight Mixing  |                      |
|  | - Embeddings   |                  | - Speed Control  |                      |
|  +----------------+                  +------------------+                      |
|         ↓                                    ↓                                 |
|    Dynamic Loading                    Optimized Processing                     |
|    & Hot-swapping                    & Memory Management                       |
+--------------------------------------------------------------------------------+
```

The voice generation system uses tensor operations for voice mixing, allowing for weighted combinations of different voice characteristics. This enables dynamic voice transitions and the creation of personalized voices by mixing existing voice embeddings.

## Natural Interruption System

The system includes an interruption detection system that uses a continuous feedback loop. This loop monitors for speech input while processing and playing audio output. This creates a natural conversation flow where users can interrupt the AI's response at any time, similar to a human conversation.

```ascii
                           Interruption System
+--------------------------------------------------------------------------------+
|                                                                                |
|    Speech Monitor              Interrupt Handler           Queue Manager       |
|  +--------------+            +-----------------+         +--------------+      |
|  | Continuous   |            | Audio Stream    |         | Clear &      |      |
|  | VAD Check    |----------->| Termination     |-------->| Reset        |      |
|  +--------------+            +-----------------+         +--------------+      |
|         ↓                            ↓                          ↓              |
|    Threshold-based              Graceful                   State               |
|    Detection                    Shutdown                   Recovery            |
+--------------------------------------------------------------------------------+
```

The interrupt handling system is integrated into the audio processing pipeline. It uses queue management to immediately respond to new speech input while maintaining the conversation's context.

## Technical Implementation

The system is built using Python 3.8+ (3.12 was used for testing). Core processing uses PyTorch for tensor operations and model inference. Real-time audio processing is handled using PyAudio and sounddevice libraries. The speech recognition component uses Whisper for transcription, and the system uses CUDA acceleration when available.

The implementation includes queue management for data flow, with separate queues for text processing and audio generation. This multi-threaded architecture ensures that computationally intensive tasks do not impact system responsiveness. The interrupt handling system enables a natural conversation flow by immediately responding to user input.

Through optimization of the audio processing pipeline and memory management, the system maintains low latency while delivering voice synthesis. The modular architecture allows for extension and modification of individual components.

## Configuration System

The system uses a configuration management system that allows for customization of parameters:

- Voice model and embedding settings
- Audio processing parameters
- VAD sensitivity thresholds
- Language model configurations
- System paths and environment settings

All parameters can be adjusted through environment variables or the configuration file, allowing for performance tuning.
