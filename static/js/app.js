// Nirmaan AI Communication Scorer - Frontend JavaScript

class CommunicationScorer {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.sampleData = {
            transcript: `Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School. I am 13 years old. I live with my family. There are 3 people in my family, me, my mother and my father. One special thing about my family is that they are very kind hearted to everyone and soft spoken. One thing I really enjoy is play, playing cricket and taking wickets. A fun fact about me is that I see movies and talk by myself. One thing people don't know about me is that I once stole a toy from one of my cousin. My favourite subject is science because it is very interesting. Through science I can explore the whole world and make the discoveries and improve the lives of others. Thank you for listening.`,
            duration: 52
        };
    }

    initializeElements() {
        this.form = document.getElementById('scoringForm');
        this.transcriptInput = document.getElementById('transcript');
        this.durationInput = document.getElementById('duration');
        this.scoreBtn = document.getElementById('scoreBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.loadSampleBtn = document.getElementById('loadSampleBtn');
        this.createSampleEmbeddingsBtn = document.getElementById('createSampleEmbeddingsBtn');
        this.findSimilarBtn = document.getElementById('findSimilarBtn');
        this.resultsContainer = document.getElementById('resultsContainer');
        this.noResults = document.getElementById('noResults');
        this.detailedResultsRow = document.getElementById('detailedResultsRow');
        this.detailedResults = document.getElementById('detailedResults');
        this.wordCount = document.getElementById('wordCount');
        this.charCount = document.getElementById('charCount');
        this.loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
        
        // Audio recording elements
        this.startRecordingBtn = document.getElementById('startRecordingBtn');
        this.stopRecordingBtn = document.getElementById('stopRecordingBtn');
        this.playRecordingBtn = document.getElementById('playRecordingBtn');
        this.clearRecordingBtn = document.getElementById('clearRecordingBtn');
        this.recordingTimer = document.getElementById('recordingTimer');
        this.timerDisplay = document.getElementById('timerDisplay');
        this.recordingProgress = document.getElementById('recordingProgress');
        this.audioPlayback = document.getElementById('audioPlayback');
        this.recordingStatus = document.getElementById('recordingStatus');
        
        // Recording state
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.recordingStartTime = null;
        this.recordingInterval = null;
        this.recordedBlob = null;
    }

    bindEvents() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.clearBtn.addEventListener('click', () => this.clearForm());
        this.loadSampleBtn.addEventListener('click', () => this.loadSampleData());
        this.createSampleEmbeddingsBtn.addEventListener('click', () => this.createSampleEmbeddings());
        this.findSimilarBtn.addEventListener('click', () => this.findSimilarTexts());
        this.transcriptInput.addEventListener('input', () => this.updateCounts());
        
        // Audio recording event listeners
        this.startRecordingBtn.addEventListener('click', () => this.startRecording());
        this.stopRecordingBtn.addEventListener('click', () => this.stopRecording());
        this.playRecordingBtn.addEventListener('click', () => this.playRecording());
        this.clearRecordingBtn.addEventListener('click', () => this.clearRecording());
    }

    updateCounts() {
        const text = this.transcriptInput.value;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        const chars = text.length;
        
        this.wordCount.textContent = `${words} words`;
        this.charCount.textContent = `${chars} characters`;
    }

    loadSampleData() {
        this.transcriptInput.value = this.sampleData.transcript;
        this.durationInput.value = this.sampleData.duration;
        this.updateCounts();
        
        // Show a brief notification
        this.showNotification('Sample data loaded successfully!', 'success');
    }

    clearForm() {
        this.form.reset();
        this.updateCounts();
        this.hideResults();
        this.findSimilarBtn.disabled = true;
        this.showNotification('Form cleared', 'info');
    }

    async createSampleEmbeddings() {
        this.showLoading('Creating sample embeddings...');
        
        try {
            const response = await fetch('/api/sample-embeddings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.showNotification('Sample embeddings created successfully!', 'success');
            
            // Show embeddings info
            this.displayEmbeddingsInfo(result.result);
            
        } catch (error) {
            console.error('Error creating sample embeddings:', error);
            this.showNotification('Failed to create sample embeddings', 'error');
        } finally {
            this.hideLoading();
        }
    }

    // Audio Recording Methods
    async startRecording() {
        try {
            // Request microphone permission
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000
                } 
            });
            
            // Force WAV format for better Azure Speech Services compatibility
            // Use Web Audio API to capture and convert to WAV
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000 // Azure Speech Services optimal sample rate
            });
            
            this.sourceNode = this.audioContext.createMediaStreamSource(stream);
            this.processorNode = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            // Store stream reference for cleanup
            this.currentStream = stream;
            this.audioBuffers = [];
            
            this.processorNode.onaudioprocess = (event) => {
                if (!this.isRecording) return;
                
                const inputBuffer = event.inputBuffer;
                const inputData = inputBuffer.getChannelData(0);
                
                // Convert Float32Array to Int16Array for WAV format
                const int16Data = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    int16Data[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                }
                
                this.audioBuffers.push(int16Data);
            };
            
            // Connect nodes
            this.sourceNode.connect(this.processorNode);
            this.processorNode.connect(this.audioContext.destination);
            
            // Start recording
            this.recordingStartTime = Date.now();
            this.isRecording = true;
            
            // Update UI
            this.updateRecordingUI(true);
            this.startRecordingTimer();
            
            this.showNotification('Recording started! Speak clearly into your microphone.', 'info');
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.showNotification('Could not access microphone. Please check permissions.', 'danger');
        }
    }
    
    stopRecording() {
        if (this.isRecording) {
            this.isRecording = false;
            
            // Stop audio processing
            if (this.processorNode) {
                this.processorNode.disconnect();
                this.sourceNode.disconnect();
            }
            
            // Stop all tracks to release microphone
            if (this.currentStream) {
                this.currentStream.getTracks().forEach(track => track.stop());
            }
            
            // Create WAV file from collected audio buffers
            this.createWavFile();
            
            this.stopRecordingTimer();
            this.updateRecordingUI(false);
            
            this.showNotification('Recording stopped! Processing audio...', 'success');
        }
    }
    
    createWavFile() {
        if (this.audioBuffers.length === 0) {
            console.error('No audio data recorded');
            return;
        }
        
        // Calculate total length
        let totalLength = 0;
        for (const buffer of this.audioBuffers) {
            totalLength += buffer.length;
        }
        
        // Combine all buffers
        const combinedBuffer = new Int16Array(totalLength);
        let offset = 0;
        for (const buffer of this.audioBuffers) {
            combinedBuffer.set(buffer, offset);
            offset += buffer.length;
        }
        
        // Create WAV file
        const wavBuffer = this.encodeWAV(combinedBuffer, 16000, 1);
        this.recordedBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        
        // Clear buffers
        this.audioBuffers = [];
        
        // Auto-process the recording
        this.onRecordingComplete();
    }
    
    encodeWAV(samples, sampleRate, numChannels) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        
        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numChannels * 2, true);
        view.setUint16(32, numChannels * 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);
        
        // Write PCM data
        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            view.setInt16(offset, samples[i], true);
            offset += 2;
        }
        
        return buffer;
    }
    
    playRecording() {
        if (this.recordedBlob) {
            const audioUrl = URL.createObjectURL(this.recordedBlob);
            this.audioPlayback.src = audioUrl;
            this.audioPlayback.classList.remove('d-none');
            this.audioPlayback.play();
        }
    }
    
    clearRecording() {
        this.recordedBlob = null;
        this.recordedChunks = [];
        this.audioPlayback.classList.add('d-none');
        this.audioPlayback.src = '';
        
        // Reset UI
        this.playRecordingBtn.classList.add('d-none');
        this.clearRecordingBtn.classList.add('d-none');
        this.recordingStatus.innerHTML = `
            <i class="fas fa-microphone fa-3x text-muted mb-2"></i>
            <p class="mb-0 text-muted">Click to start recording your self-introduction</p>
        `;
        
        this.showNotification('Recording cleared', 'info');
    }
    
    updateRecordingUI(isRecording) {
        if (isRecording) {
            this.startRecordingBtn.classList.add('d-none');
            this.stopRecordingBtn.classList.remove('d-none');
            this.recordingTimer.classList.remove('d-none');
            
            // Add recording visual feedback
            const recordingCard = document.querySelector('.card.border-dashed');
            recordingCard.classList.add('recording-active');
            
            this.recordingStatus.innerHTML = `
                <i class="fas fa-microphone fa-3x text-danger mb-2"></i>
                <p class="mb-0 text-danger fw-bold">Recording in progress...</p>
            `;
            
        } else {
            this.startRecordingBtn.classList.remove('d-none');
            this.stopRecordingBtn.classList.add('d-none');
            this.recordingTimer.classList.add('d-none');
            
            // Remove recording visual feedback
            const recordingCard = document.querySelector('.card.border-dashed');
            recordingCard.classList.remove('recording-active');
            
            if (this.recordedBlob) {
                this.playRecordingBtn.classList.remove('d-none');
                this.clearRecordingBtn.classList.remove('d-none');
                
                this.recordingStatus.innerHTML = `
                    <i class="fas fa-check-circle fa-3x text-success mb-2"></i>
                    <p class="mb-0 text-success fw-bold">Recording completed!</p>
                `;
            }
        }
    }
    
    startRecordingTimer() {
        this.recordingInterval = setInterval(() => {
            const elapsed = Date.now() - this.recordingStartTime;
            const seconds = Math.floor(elapsed / 1000);
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            
            this.timerDisplay.textContent = 
                `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
            
            // Update progress bar (max 2 minutes)
            const maxDuration = 120; // 2 minutes
            const progress = Math.min((seconds / maxDuration) * 100, 100);
            this.recordingProgress.style.width = `${progress}%`;
            
            // Auto-stop after 2 minutes
            if (seconds >= maxDuration) {
                this.stopRecording();
                this.showNotification('Recording stopped automatically after 2 minutes', 'warning');
            }
        }, 1000);
    }
    
    stopRecordingTimer() {
        if (this.recordingInterval) {
            clearInterval(this.recordingInterval);
            this.recordingInterval = null;
        }
    }
    
    async onRecordingComplete() {
        try {
            // Auto-process the recording
            await this.processRecording();
        } catch (error) {
            console.error('Error processing recording:', error);
            this.showNotification('Error processing recording. Please try again.', 'danger');
        }
    }
    
    async processRecording() {
        if (!this.recordedBlob) {
            this.showNotification('No recording available to process', 'warning');
            return;
        }
        
        try {
            this.showLoadingModal('Processing recording and generating embeddings...');
            
            // Create FormData with the recorded audio (always WAV now)
            const formData = new FormData();
            formData.append('audio', this.recordedBlob, 'recording.wav');
            
            const response = await fetch('/api/record-and-process', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Update transcript input with the transcribed text
                this.transcriptInput.value = data.transcript;
                this.updateCounts();
                
                // Auto-fill duration if provided
                if (data.duration) {
                    this.durationInput.value = Math.round(data.duration);
                }
                
                this.showNotification(
                    `Recording processed successfully! ${data.transcript.split(' ').length} words transcribed.`, 
                    'success'
                );
                
                // Show Azure Storage info
                if (data.storage_info) {
                    this.showNotification(
                        `Audio stored in Azure Storage: ${data.storage_info.blob_name}`, 
                        'info'
                    );
                }
                
                // Show embeddings info
                if (data.embeddings_info && data.embeddings_info.success) {
                    this.showNotification(
                        `Vector embeddings created and stored! ID: ${data.embeddings_info.embeddings_id}`, 
                        'success'
                    );
                }
                
            } else {
                this.showNotification(`Processing failed: ${data.error}`, 'danger');
            }
            
        } catch (error) {
            console.error('Error processing recording:', error);
            this.showNotification('Error processing recording. Please check your connection and try again.', 'danger');
        } finally {
            this.hideLoadingModal();
        }
    }

    async findSimilarTexts() {
        const text = this.transcriptInput.value.trim();
        if (!text) {
            this.showNotification('Please enter text first', 'error');
            return;
        }

        this.showLoading('Finding similar texts...');
        
        try {
            const response = await fetch('/api/similar', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    top_k: 5
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displaySimilarTexts(result);
            
        } catch (error) {
            console.error('Error finding similar texts:', error);
            this.showNotification('Failed to find similar texts', 'error');
        } finally {
            this.hideLoading();
        }
    }

    handleAudioUpload() {
        const audioFile = this.audioFileInput.files[0];
        if (audioFile) {
            // Show file name and enable transcription
            this.showNotification(`Audio file selected: ${audioFile.name}`, 'info');
            
            // Automatically start transcription
            this.uploadAndTranscribeAudio();
        }
    }

    async uploadAndTranscribeAudio() {
        const audioFile = this.audioFileInput.files[0];
        if (!audioFile) {
            this.showNotification('Please select an audio file first', 'warning');
            return;
        }

        // Validate file size (max 25MB)
        const maxSize = 25 * 1024 * 1024; // 25MB
        if (audioFile.size > maxSize) {
            this.showNotification('Audio file is too large. Maximum size is 25MB.', 'danger');
            return;
        }

        // Validate file format - Azure Speech Services supported formats
        const supportedFormats = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'mp4', 'wma'];
        const fileExtension = audioFile.name.split('.').pop().toLowerCase();
        if (!supportedFormats.includes(fileExtension)) {
            this.showNotification(
                `Unsupported audio format: .${fileExtension}. Azure Speech Services supports: ${supportedFormats.join(', ')}`, 
                'danger'
            );
            return;
        }

        const formData = new FormData();
        formData.append('audio', audioFile);

        try {
            this.showLoadingModal(`Transcribing ${fileExtension.toUpperCase()} audio file...`);
            
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (data.success) {
                this.transcriptInput.value = data.transcript;
                this.updateCounts();
                this.showNotification(
                    `Audio transcribed successfully! ${data.transcript.split(' ').length} words detected.`, 
                    'success'
                );
                
                // Show embeddings info if available
                if (data.embeddings_info && data.embeddings_info.success) {
                    this.showNotification(
                        `Embeddings created! ID: ${data.embeddings_info.embeddings_id}`, 
                        'info'
                    );
                }
            } else {
                let errorMessage = `Transcription failed: ${data.error}`;
                
                // Provide specific guidance for common issues
                if (data.error.includes('SPXERR_INVALID_HEADER') || data.error.includes('format')) {
                    errorMessage += `
                    
                    💡 Try: Ensure your audio file is not corrupted and is in a supported format (WAV, MP3, OGG, FLAC, AAC, MP4, WMA).`;
                }
                
                this.showNotification(errorMessage, 'danger');
            }
        } catch (error) {
            console.error('Error transcribing audio:', error);
            this.showNotification('Error transcribing audio. Please check your internet connection and try again.', 'danger');
        } finally {
            this.hideLoadingModal();
        }
    }

    async handleSubmit(e) {
        e.preventDefault();
        
        const transcript = this.transcriptInput.value.trim();
        const duration = this.durationInput.value ? parseInt(this.durationInput.value) : null;
        
        if (!transcript) {
            this.showNotification('Please enter a transcript', 'error');
            return;
        }

        this.showLoading();
        
        try {
            const response = await fetch('/api/score', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    transcript: transcript,
                    duration_seconds: duration
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const results = await response.json();
            this.displayResults(results);
            
        } catch (error) {
            console.error('Error scoring transcript:', error);
            this.showError('Failed to score transcript. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    createEmbeddingsInfoHtml(embeddingsInfo) {
        if (!embeddingsInfo.success) {
            return `
                <div class="mt-3">
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        <strong>Embeddings:</strong> ${embeddingsInfo.error || 'Failed to create embeddings'}
                    </div>
                </div>
            `;
        }

        let similarityHtml = '';
        if (embeddingsInfo.similarity_matches && embeddingsInfo.similarity_matches.length > 0) {
            similarityHtml = `
                <div class="mt-2">
                    <h6>Similar Texts Found:</h6>
                    <ul class="list-unstyled">
                        ${embeddingsInfo.similarity_matches.map(match => `
                            <li class="mb-1">
                                <small class="text-muted">
                                    Similarity: ${(match.similarity * 100).toFixed(1)}% - 
                                    ${match.text_preview}
                                </small>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            `;
        }

        return `
            <div class="mt-3">
                <div class="alert alert-success">
                    <i class="fas fa-vector-square me-2"></i>
                    <strong>Embeddings Created:</strong> 
                    ID: ${embeddingsInfo.embeddings_id}<br>
                    <small>Dimension: ${embeddingsInfo.embeddings_dimension} | Stored in Azure Cosmos DB</small>
                    ${similarityHtml}
                </div>
            </div>
        `;
    }

    displayEmbeddingsInfo(embeddingsInfo) {
        const embeddingsHtml = this.createEmbeddingsInfoHtml(embeddingsInfo);
        
        // Add to results container if it exists, otherwise create a temporary display
        if (this.resultsContainer.innerHTML.trim() === '') {
            this.hideNoResults();
            this.resultsContainer.innerHTML = embeddingsHtml;
        } else {
            this.resultsContainer.innerHTML += embeddingsHtml;
        }
    }

    displaySimilarTexts(result) {
        const similarTextsHtml = `
            <div class="mt-4">
                <h5><i class="fas fa-search me-2"></i>Similar Texts</h5>
                <div class="alert alert-info">
                    <strong>Query:</strong> ${result.query_text.substring(0, 100)}...
                </div>
                ${result.similar_texts.length === 0 ? 
                    '<p class="text-muted">No similar texts found.</p>' :
                    result.similar_texts.map((item, index) => `
                        <div class="card mb-2">
                            <div class="card-body p-3">
                                <div class="d-flex justify-content-between align-items-start mb-2">
                                    <h6 class="mb-0">Match ${index + 1}</h6>
                                    <span class="badge bg-primary">${(item.similarity * 100).toFixed(1)}% similar</span>
                                </div>
                                <p class="mb-1">${item.document.text.substring(0, 200)}...</p>
                                <small class="text-muted">
                                    Created: ${new Date(item.document.created_at).toLocaleString()}
                                    ${item.document.metadata && item.document.metadata.student_name ? 
                                        ` | Student: ${item.document.metadata.student_name}` : ''}
                                </small>
                            </div>
                        </div>
                    `).join('')
                }
            </div>
        `;
        
        // Add to detailed results
        this.detailedResults.innerHTML += similarTextsHtml;
        this.detailedResultsRow.style.display = 'block';
    }

    showLoading(message = 'Analyzing...') {
        this.scoreBtn.disabled = true;
        this.scoreBtn.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${message}`;
        this.loadingModal.show();
    }

    hideLoading() {
        this.scoreBtn.disabled = false;
        this.scoreBtn.innerHTML = '<i class="fas fa-chart-line me-2"></i>Score Communication';
        this.loadingModal.hide();
    }

    displayResults(results) {
        this.hideNoResults();
        
        // Display overall score
        const overallScoreHtml = this.createOverallScoreHtml(results.overall_score, results.word_count);
        
        // Display criteria breakdown
        const criteriaHtml = this.createCriteriaBreakdownHtml(results.criteria_scores);
        
        // Display embeddings info if available
        let embeddingsHtml = '';
        if (results.embeddings_info) {
            embeddingsHtml = this.createEmbeddingsInfoHtml(results.embeddings_info);
            // Enable find similar button if embeddings were created successfully
            if (results.embeddings_info.success) {
                this.findSimilarBtn.disabled = false;
            }
        }
        
        this.resultsContainer.innerHTML = overallScoreHtml + criteriaHtml + embeddingsHtml;
        
        // Display detailed feedback
        this.displayDetailedResults(results.detailed_feedback);
        
        // Add animation
        this.resultsContainer.classList.add('fade-in-up');
        
        // Show detailed results section
        this.detailedResultsRow.style.display = 'block';
        
        // Scroll to results
        this.resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    createOverallScoreHtml(score, wordCount) {
        const scoreClass = this.getScoreClass(score);
        const scoreLabel = this.getScoreLabel(score);
        
        return `
            <div class="results-header">
                <div class="score-circle ${scoreClass}">
                    <span>${score}</span>
                </div>
                <h4 class="mb-2">Overall Score: ${score}/100</h4>
                <p class="text-muted mb-3">
                    <span class="score-label">${scoreLabel}</span> | ${wordCount} words
                </p>
                <div class="progress mb-3">
                    <div class="progress-bar ${this.getProgressBarClass(score)}" 
                         role="progressbar" 
                         style="width: ${score}%" 
                         aria-valuenow="${score}" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                    </div>
                </div>
            </div>
        `;
    }

    createCriteriaBreakdownHtml(criteriaScores) {
        const criteriaLabels = {
            content_structure: 'Content & Structure',
            speech_rate: 'Speech Rate',
            language_grammar: 'Language & Grammar',
            clarity: 'Clarity',
            engagement: 'Engagement'
        };

        const criteriaWeights = {
            content_structure: 40,
            speech_rate: 10,
            language_grammar: 20,
            clarity: 15,
            engagement: 15
        };

        let html = '<div class="row">';
        
        Object.entries(criteriaScores).forEach(([key, score]) => {
            const label = criteriaLabels[key] || key;
            const weight = criteriaWeights[key] || 0;
            const percentage = weight > 0 ? Math.round((score / weight) * 100) : 0;
            const badgeClass = this.getBadgeClass(percentage);
            
            html += `
                <div class="col-md-6 col-lg-4 mb-3">
                    <div class="card criterion-card ${this.getScoreClass(percentage)}">
                        <div class="card-body p-3">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h6 class="mb-0">${label}</h6>
                                <span class="badge ${badgeClass}">${score}/${weight}</span>
                            </div>
                            <div class="progress" style="height: 6px;">
                                <div class="progress-bar ${this.getProgressBarClass(percentage)}" 
                                     style="width: ${percentage}%"></div>
                            </div>
                            <small class="text-muted">${percentage}%</small>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        return html;
    }

    displayDetailedResults(detailedFeedback) {
        let html = '<div class="accordion" id="feedbackAccordion">';
        
        const criteriaInfo = {
            content_structure: {
                title: 'Content & Structure Analysis',
                icon: 'fas fa-structure',
                color: 'primary'
            },
            speech_rate: {
                title: 'Speech Rate Analysis',
                icon: 'fas fa-tachometer-alt',
                color: 'info'
            },
            language_grammar: {
                title: 'Language & Grammar Analysis',
                icon: 'fas fa-spell-check',
                color: 'success'
            },
            clarity: {
                title: 'Clarity Analysis',
                icon: 'fas fa-eye',
                color: 'warning'
            },
            engagement: {
                title: 'Engagement Analysis',
                icon: 'fas fa-heart',
                color: 'danger'
            }
        };

        Object.entries(detailedFeedback).forEach(([key, feedback], index) => {
            const info = criteriaInfo[key] || { title: key, icon: 'fas fa-info', color: 'secondary' };
            const isExpanded = index === 0 ? 'show' : '';
            const isCollapsed = index === 0 ? '' : 'collapsed';
            
            html += `
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading${index}">
                        <button class="accordion-button ${isCollapsed}" type="button" 
                                data-bs-toggle="collapse" data-bs-target="#collapse${index}" 
                                aria-expanded="${index === 0}" aria-controls="collapse${index}">
                            <i class="${info.icon} me-2 text-${info.color}"></i>
                            ${info.title}
                        </button>
                    </h2>
                    <div id="collapse${index}" class="accordion-collapse collapse ${isExpanded}" 
                         aria-labelledby="heading${index}" data-bs-parent="#feedbackAccordion">
                        <div class="accordion-body">
                            ${this.formatFeedbackContent(feedback)}
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        this.detailedResults.innerHTML = html;
    }

    formatFeedbackContent(feedback) {
        if (typeof feedback === 'string') {
            return `<p class="feedback-text">${feedback}</p>`;
        }
        
        let html = '';
        
        // Handle enhanced feedback structure
        if (feedback.overall_analysis) {
            return this.formatEnhancedAnalysis(feedback.overall_analysis);
        }
        
        // Handle speech rate analysis
        if (feedback.current_rate) {
            return this.formatSpeechRateAnalysis(feedback);
        }
        
        // Handle language analysis
        if (feedback.vocabulary_richness) {
            return this.formatLanguageAnalysis(feedback);
        }
        
        // Handle clarity analysis
        if (feedback.filler_words_count !== undefined) {
            return this.formatClarityAnalysis(feedback);
        }
        
        // Handle engagement analysis
        if (feedback.sentiment_score) {
            return this.formatEngagementAnalysis(feedback);
        }
        
        // Legacy format handling
        if (feedback.score !== undefined) {
            html += `<div class="mb-2"><strong>Score:</strong> ${feedback.score}</div>`;
        }
        
        if (feedback.feedback) {
            html += `<div class="feedback-text">${feedback.feedback.replace(/\n/g, '<br>')}</div>`;
        }
        
        // Handle nested feedback (like content_structure)
        Object.entries(feedback).forEach(([key, value]) => {
            if (key !== 'score' && key !== 'feedback' && typeof value === 'object') {
                html += `
                    <div class="mt-3">
                        <h6 class="text-capitalize">${key.replace('_', ' ')}</h6>
                        <div class="ms-3">
                            ${this.formatFeedbackContent(value)}
                        </div>
                    </div>
                `;
            }
        });
        
        return html || '<p class="text-muted">No detailed feedback available</p>';
    }
    
    formatEnhancedAnalysis(analysis) {
        let html = `
            <div class="enhanced-analysis">
                <div class="score-header mb-3">
                    <h5 class="d-flex justify-content-between align-items-center">
                        <span>Performance: ${analysis.performance_level}</span>
                        <span class="badge bg-primary">${analysis.score_breakdown}</span>
                    </h5>
                </div>
        `;
        
        // Strengths
        if (analysis.strengths && analysis.strengths.length > 0) {
            html += `
                <div class="mb-3">
                    <h6 class="text-success"><i class="fas fa-check-circle me-2"></i>Strengths</h6>
                    <ul class="list-unstyled ms-3">
                        ${analysis.strengths.map(strength => `<li class="text-success mb-1">${strength}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Improvements
        if (analysis.improvements && analysis.improvements.length > 0) {
            html += `
                <div class="mb-3">
                    <h6 class="text-warning"><i class="fas fa-exclamation-triangle me-2"></i>Areas for Improvement</h6>
                    <ul class="list-unstyled ms-3">
                        ${analysis.improvements.map(improvement => `<li class="text-warning mb-1">${improvement}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Examples
        if (analysis.examples && Object.keys(analysis.examples).length > 0) {
            html += `
                <div class="mb-3">
                    <h6 class="text-info"><i class="fas fa-lightbulb me-2"></i>Examples & Suggestions</h6>
                    <div class="ms-3">
            `;
            
            Object.entries(analysis.examples).forEach(([key, example]) => {
                const title = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                html += `
                    <div class="example-item mb-2 p-2 bg-light rounded">
                        <strong>${title}:</strong> ${example}
                    </div>
                `;
            });
            
            html += `</div></div>`;
        }
        
        html += `</div>`;
        return html;
    }
    
    formatSpeechRateAnalysis(analysis) {
        return `
            <div class="speech-rate-analysis">
                <div class="score-header mb-3">
                    <h5 class="d-flex justify-content-between align-items-center">
                        <span>${analysis.current_rate}</span>
                        <span class="badge bg-info">${analysis.score}</span>
                    </h5>
                    <p class="text-muted mb-0">Optimal Range: ${analysis.optimal_range}</p>
                </div>
                
                <div class="assessment mb-3">
                    <h6 class="text-primary"><i class="fas fa-chart-line me-2"></i>Assessment</h6>
                    <p class="ms-3">${analysis.assessment}</p>
                </div>
                
                ${analysis.suggestions && analysis.suggestions.length > 0 ? `
                    <div class="suggestions">
                        <h6 class="text-success"><i class="fas fa-tips me-2"></i>Suggestions</h6>
                        <ul class="list-unstyled ms-3">
                            ${analysis.suggestions.map(suggestion => `<li class="mb-1">${suggestion}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    formatLanguageAnalysis(analysis) {
        let html = `
            <div class="language-analysis">
                <div class="score-header mb-3">
                    <div class="row">
                        <div class="col-md-4">
                            <strong>Total Score:</strong> ${analysis.total_score}
                        </div>
                        <div class="col-md-4">
                            <strong>Vocabulary:</strong> ${analysis.ttr_score}
                        </div>
                        <div class="col-md-4">
                            <strong>Grammar:</strong> ${analysis.grammar_score}
                        </div>
                    </div>
                    <p class="text-muted mt-2">${analysis.vocabulary_richness}</p>
                </div>
        `;
        
        // Strengths and improvements
        if (analysis.strengths && analysis.strengths.length > 0) {
            html += `
                <div class="mb-3">
                    <h6 class="text-success"><i class="fas fa-check-circle me-2"></i>Strengths</h6>
                    <ul class="list-unstyled ms-3">
                        ${analysis.strengths.map(strength => `<li class="text-success mb-1">${strength}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        if (analysis.improvements && analysis.improvements.length > 0) {
            html += `
                <div class="mb-3">
                    <h6 class="text-warning"><i class="fas fa-exclamation-triangle me-2"></i>Improvements</h6>
                    <ul class="list-unstyled ms-3">
                        ${analysis.improvements.map(improvement => `<li class="text-warning mb-1">${improvement}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        // Examples
        if (analysis.examples && Object.keys(analysis.examples).length > 0) {
            html += `
                <div class="examples">
                    <h6 class="text-info"><i class="fas fa-lightbulb me-2"></i>Examples</h6>
                    <div class="ms-3">
            `;
            
            Object.entries(analysis.examples).forEach(([key, example]) => {
                const title = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                html += `
                    <div class="example-item mb-2 p-2 bg-light rounded">
                        <strong>${title}:</strong> ${example}
                    </div>
                `;
            });
            
            html += `</div></div>`;
        }
        
        html += `</div>`;
        return html;
    }
    
    formatClarityAnalysis(analysis) {
        return `
            <div class="clarity-analysis">
                <div class="score-header mb-3">
                    <h5 class="d-flex justify-content-between align-items-center">
                        <span>Filler Rate: ${analysis.filler_rate}</span>
                        <span class="badge bg-warning">${analysis.score}</span>
                    </h5>
                    <p class="text-muted mb-0">Filler Words Count: ${analysis.filler_words_count}</p>
                </div>
                
                <div class="assessment mb-3">
                    <h6 class="text-primary"><i class="fas fa-eye me-2"></i>Assessment</h6>
                    <p class="ms-3">${analysis.assessment}</p>
                </div>
                
                ${this.formatStrengthsAndImprovements(analysis)}
                ${this.formatExamples(analysis.examples)}
            </div>
        `;
    }
    
    formatEngagementAnalysis(analysis) {
        return `
            <div class="engagement-analysis">
                <div class="score-header mb-3">
                    <h5 class="d-flex justify-content-between align-items-center">
                        <span>Sentiment: ${analysis.sentiment_score}</span>
                        <span class="badge bg-danger">${analysis.score}</span>
                    </h5>
                    <p class="text-muted mb-0">Range: ${analysis.sentiment_range}</p>
                </div>
                
                <div class="assessment mb-3">
                    <h6 class="text-primary"><i class="fas fa-heart me-2"></i>Assessment</h6>
                    <p class="ms-3">${analysis.assessment}</p>
                </div>
                
                ${this.formatStrengthsAndImprovements(analysis)}
                ${this.formatExamples(analysis.examples)}
            </div>
        `;
    }
    
    formatStrengthsAndImprovements(analysis) {
        let html = '';
        
        if (analysis.strengths && analysis.strengths.length > 0) {
            html += `
                <div class="mb-3">
                    <h6 class="text-success"><i class="fas fa-check-circle me-2"></i>Strengths</h6>
                    <ul class="list-unstyled ms-3">
                        ${analysis.strengths.map(strength => `<li class="text-success mb-1">${strength}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        if (analysis.improvements && analysis.improvements.length > 0) {
            html += `
                <div class="mb-3">
                    <h6 class="text-warning"><i class="fas fa-exclamation-triangle me-2"></i>Improvements</h6>
                    <ul class="list-unstyled ms-3">
                        ${analysis.improvements.map(improvement => `<li class="text-warning mb-1">${improvement}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        return html;
    }
    
    formatExamples(examples) {
        if (!examples || Object.keys(examples).length === 0) {
            return '';
        }
        
        let html = `
            <div class="examples">
                <h6 class="text-info"><i class="fas fa-lightbulb me-2"></i>Examples & Tips</h6>
                <div class="ms-3">
        `;
        
        Object.entries(examples).forEach(([key, example]) => {
            const title = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            
            if (Array.isArray(example)) {
                html += `
                    <div class="example-item mb-2 p-2 bg-light rounded">
                        <strong>${title}:</strong>
                        <ul class="mb-0 mt-1">
                            ${example.map(item => `<li>${item}</li>`).join('')}
                        </ul>
                    </div>
                `;
            } else {
                html += `
                    <div class="example-item mb-2 p-2 bg-light rounded">
                        <strong>${title}:</strong> ${example}
                    </div>
                `;
            }
        });
        
        html += `</div></div>`;
        return html;
    }

    getScoreClass(score) {
        if (score >= 85) return 'excellent';
        if (score >= 70) return 'good';
        if (score >= 50) return 'average';
        return 'poor';
    }

    getScoreLabel(score) {
        if (score >= 85) return 'Excellent';
        if (score >= 70) return 'Good';
        if (score >= 50) return 'Average';
        return 'Needs Improvement';
    }

    getProgressBarClass(score) {
        if (score >= 85) return 'bg-success';
        if (score >= 70) return 'bg-info';
        if (score >= 50) return 'bg-warning';
        return 'bg-danger';
    }

    getBadgeClass(score) {
        if (score >= 85) return 'bg-success';
        if (score >= 70) return 'bg-info';
        if (score >= 50) return 'bg-warning';
        return 'bg-danger';
    }

    hideNoResults() {
        this.noResults.style.display = 'none';
    }

    hideResults() {
        this.noResults.style.display = 'block';
        this.resultsContainer.innerHTML = '';
        this.detailedResultsRow.style.display = 'none';
    }

    showError(message) {
        this.resultsContainer.innerHTML = `
            <div class="error-state text-center">
                <i class="fas fa-exclamation-triangle fa-2x mb-3"></i>
                <h5>Error</h5>
                <p>${message}</p>
            </div>
        `;
        this.hideNoResults();
    }

    showLoadingModal(message = 'Processing...') {
        // Show loading modal if it exists
        if (this.loadingModal) {
            const modalBody = document.querySelector('#loadingModal .modal-body');
            if (modalBody) {
                modalBody.innerHTML = `
                    <div class="text-center">
                        <div class="spinner-border text-primary mb-3" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mb-0">${message}</p>
                    </div>
                `;
            }
            this.loadingModal.show();
        }
    }

    hideLoadingModal() {
        if (this.loadingModal) {
            this.loadingModal.hide();
        }
    }

    showNotification(message, type = 'info') {
        // Create a toast notification
        const toastHtml = `
            <div class="toast align-items-center text-white bg-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-${type === 'error' ? 'exclamation-circle' : type === 'success' ? 'check-circle' : 'info-circle'} me-2"></i>
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>
        `;
        
        // Create toast container if it doesn't exist
        let toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
            toastContainer.style.zIndex = '1055';
            document.body.appendChild(toastContainer);
        }
        
        // Add toast to container
        const toastElement = document.createElement('div');
        toastElement.innerHTML = toastHtml;
        toastContainer.appendChild(toastElement.firstElementChild);
        
        // Initialize and show toast
        const toast = new bootstrap.Toast(toastContainer.lastElementChild, {
            autohide: true,
            delay: 3000
        });
        toast.show();
        
        // Remove toast element after it's hidden
        toastContainer.lastElementChild.addEventListener('hidden.bs.toast', function() {
            this.remove();
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    const scorer = new CommunicationScorer();
    
    // Initialize word count
    scorer.updateCounts();
    
    console.log('Nirmaan AI Communication Scorer initialized successfully');
});
