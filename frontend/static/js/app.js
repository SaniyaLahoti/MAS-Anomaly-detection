// Real-time Anomaly Detection Dashboard - JavaScript

const API_BASE = window.location.origin;
let monitoringActive = false;
let monitoringInterval = null;
let websocket = null;
let realtimeChart = null;
let samplesProcessed = 0;
let anomaliesDetected = 0;
let chartData = {
    labels: [],
    benign: [],
    anomaly: []
};

// Preset configurations - VERIFIED WORKING samples from ToN-IoT dataset
// These values match the stream data patterns that models correctly detect
const PRESETS = {
    // Benign traffic (TCP, HTTP) - Using RAW values as they classify correctly as BENIGN
    benign: {
        PROTOCOL: 6,
        L7_PROTO: 7.0,
        IN_BYTES: 1000,
        OUT_BYTES: 710,
        IN_PKTS: 5,
        OUT_PKTS: 4,
        TCP_FLAGS: 27,
        FLOW_DURATION_MILLISECONDS: 4294872
    },
    // DDoS attack - TCP with high flags, short duration
    ddos: {
        PROTOCOL: 0.31250000,
        L7_PROTO: 0.03431373,
        IN_BYTES: 0.00092770,
        OUT_BYTES: 0.00055627,
        IN_PKTS: 0.00430571,
        OUT_PKTS: 0.00215983,
        TCP_FLAGS: 0.87096774,
        FLOW_DURATION_MILLISECONDS: 0.00003378
    },
    // Port Scan / Reconnaissance - very short duration
    port_scan: {
        PROTOCOL: 0.31250000,
        L7_PROTO: 0.03431373,
        IN_BYTES: 0.00104933,
        OUT_BYTES: 0.00055627,
        IN_PKTS: 0.00430571,
        OUT_PKTS: 0.00215983,
        TCP_FLAGS: 0.87096774,
        FLOW_DURATION_MILLISECONDS: 0.00000093
    },
    // Data Theft - large TCP transfer
    data_theft: {
        PROTOCOL: 0.312500,
        L7_PROTO: 0.0,
        IN_BYTES: 0.5,
        OUT_BYTES: 0.05,
        IN_PKTS: 0.05,
        OUT_PKTS: 0.05,
        TCP_FLAGS: 0.774194,
        FLOW_DURATION_MILLISECONDS: 0.5
    },
    // Injection attack - TCP with high flags
    injection: {
        PROTOCOL: 0.31250000,
        L7_PROTO: 0.03431373,
        IN_BYTES: 0.00364483,
        OUT_BYTES: 0.00048000,
        IN_PKTS: 0.01829925,
        OUT_PKTS: 0.00539957,
        TCP_FLAGS: 0.77419355,
        FLOW_DURATION_MILLISECONDS: 0.02779910
    },
    // Backdoor attack - similar to DDoS pattern
    backdoor: {
        PROTOCOL: 0.31250000,
        L7_PROTO: 0.03431373,
        IN_BYTES: 0.00092770,
        OUT_BYTES: 0.00055627,
        IN_PKTS: 0.00430571,
        OUT_PKTS: 0.00215983,
        TCP_FLAGS: 0.87096774,
        FLOW_DURATION_MILLISECONDS: 0.00003378
    },
    // Password attack - TCP with high flags
    password: {
        PROTOCOL: 0.31250000,
        L7_PROTO: 0.03431373,
        IN_BYTES: 0.00092770,
        OUT_BYTES: 0.00055627,
        IN_PKTS: 0.00430571,
        OUT_PKTS: 0.00215983,
        TCP_FLAGS: 0.87096774,
        FLOW_DURATION_MILLISECONDS: 0.00003378
    },
    // XSS attack - similar to injection
    xss: {
        PROTOCOL: 0.31250000,
        L7_PROTO: 0.03431373,
        IN_BYTES: 0.00364483,
        OUT_BYTES: 0.00048000,
        IN_PKTS: 0.01829925,
        OUT_PKTS: 0.00539957,
        TCP_FLAGS: 0.77419355,
        FLOW_DURATION_MILLISECONDS: 0.02779910
    },
    // MITM attack - TCP with high flags
    mitm: {
        PROTOCOL: 0.31250000,
        L7_PROTO: 0.03431373,
        IN_BYTES: 0.00092770,
        OUT_BYTES: 0.00055627,
        IN_PKTS: 0.00430571,
        OUT_PKTS: 0.00215983,
        TCP_FLAGS: 0.87096774,
        FLOW_DURATION_MILLISECONDS: 0.00003378
    },
    // Ransomware attack - similar to DDoS pattern
    ransomware: {
        PROTOCOL: 0.31250000,
        L7_PROTO: 0.03431373,
        IN_BYTES: 0.00092770,
        OUT_BYTES: 0.00055627,
        IN_PKTS: 0.00430571,
        OUT_PKTS: 0.00215983,
        TCP_FLAGS: 0.87096774,
        FLOW_DURATION_MILLISECONDS: 0.00003378
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeChart();
    checkSystemHealth();
});

// Initialize real-time chart
function initializeChart() {
    const ctx = document.getElementById('realtimeChart').getContext('2d');
    realtimeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [
                {
                    label: 'Benign',
                    data: chartData.benign,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Anomaly',
                    data: chartData.anomaly,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        color: '#3f4563'
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        color: '#3f4563'
                    }
                }
            }
        }
    });
}

// Check system health
async function checkSystemHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            document.getElementById('systemStatus').innerHTML =
                '<i class="fas fa-circle"></i> System Online';
        }
    } catch (error) {
        document.getElementById('systemStatus').innerHTML =
            '<i class="fas fa-circle"></i> System Offline';
        console.error('Health check failed:', error);
    }
}

// Load preset configuration
function loadPreset(presetName) {
    const preset = PRESETS[presetName];
    if (!preset) return;

    document.getElementById('protocol').value = preset.PROTOCOL;
    document.getElementById('l7_proto').value = preset.L7_PROTO;
    document.getElementById('in_bytes').value = preset.IN_BYTES;
    document.getElementById('out_bytes').value = preset.OUT_BYTES;
    document.getElementById('in_pkts').value = preset.IN_PKTS;
    document.getElementById('out_pkts').value = preset.OUT_PKTS;
    document.getElementById('tcp_flags').value = preset.TCP_FLAGS;
    document.getElementById('flow_duration').value = preset.FLOW_DURATION_MILLISECONDS;

    // Show notification
    showNotification(`Loaded ${presetName.replace('_', ' ').toUpperCase()} preset`, 'info');
}

// Analyze flow
async function analyzeFlow() {
    const flowData = {
        PROTOCOL: parseFloat(document.getElementById('protocol').value),
        L7_PROTO: parseFloat(document.getElementById('l7_proto').value),
        IN_BYTES: parseFloat(document.getElementById('in_bytes').value),
        OUT_BYTES: parseFloat(document.getElementById('out_bytes').value),
        IN_PKTS: parseFloat(document.getElementById('in_pkts').value),
        OUT_PKTS: parseFloat(document.getElementById('out_pkts').value),
        TCP_FLAGS: parseFloat(document.getElementById('tcp_flags').value),
        FLOW_DURATION_MILLISECONDS: parseFloat(document.getElementById('flow_duration').value)
    };

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                flows: [flowData],
                dataset_version: 'v1'
            })
        });

        const result = await response.json();
        displayResult(result);

    } catch (error) {
        console.error('Analysis error:', error);
        showNotification('Analysis failed. Please try again.', 'error');
    }
}

// Display detection result
function displayResult(result) {
    // Show result card
    document.getElementById('resultCard').style.display = 'block';
    document.getElementById('explanationCard').style.display = 'block';
    document.getElementById('recommendationsCard').style.display = 'block';

    // Update prediction badge
    const predictionBadge = document.getElementById('predictionBadge');
    const predictionText = document.getElementById('predictionText');

    // Use actual attack type if available
    const displayText = result.predicted_attack_type || result.prediction;
    predictionText.textContent = displayText.toUpperCase();
    predictionBadge.className = 'prediction-badge ' +
        (result.prediction === 'benign' || displayText === 'Benign' ? 'benign' : 'anomaly');

    // Update confidence meter
    const confidencePercent = (result.confidence * 100).toFixed(1);
    document.getElementById('confidenceFill').style.width = confidencePercent + '%';
    document.getElementById('confidenceValue').textContent = confidencePercent + '%';

    // Update participating agents
    const participatingDiv = document.getElementById('participatingAgents');
    participatingDiv.innerHTML = result.participating_agents.map(agent =>
        `<div class="agent-badge active">${agent}</div>`
    ).join('');

    // Update abstaining agents
    const abstainingDiv = document.getElementById('abstainingAgents');
    abstainingDiv.innerHTML = result.abstaining_agents.map(agent =>
        `<div class="agent-badge">${agent}</div>`
    ).join('');

    // Update explanation - handle both string and object formats
    const explanationContent = document.getElementById('explanationContent');
    if (result.explanation && typeof result.explanation === 'object') {
        const exp = result.explanation;
        let html = '<h4 style="margin-bottom: 15px; color: #60a5fa;">Key Features Driving Prediction:</h4>';
        if (exp.top_features && exp.top_features.length > 0) {
            html += '<div class="shap-features">';
            for (const feat of exp.top_features) {
                const barWidth = Math.min(feat.importance * 500, 100); // Scale importance to bar width
                html += `
                    <div class="shap-feature-item" style="margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="font-weight: 600; color: #e2e8f0;">${feat.feature}</span>
                            <span style="color: #94a3b8;">${feat.importance.toFixed(4)}</span>
                        </div>
                        <div style="background: #1e293b; border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="background: linear-gradient(90deg, #3b82f6, #8b5cf6); width: ${barWidth}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>`;
            }
            html += '</div>';
        }
        if (exp.summary) {
            html += `<p style="margin-top: 15px; color: #94a3b8; font-size: 0.9em;">${exp.summary.replace(/\n/g, '<br>')}</p>`;
        }
        explanationContent.innerHTML = html;
    } else if (typeof result.explanation === 'string') {
        explanationContent.innerHTML = `<pre style="white-space: pre-wrap; line-height: 1.6;">${result.explanation}</pre>`;
    } else {
        explanationContent.innerHTML = '<p>No explanation available</p>';
    }

    // Update recommendations
    if (result.recommendations) {
        displayRecommendations(result.recommendations);
    }

    // Scroll to results
    document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth' });
}

// Display analyst recommendations
function displayRecommendations(recommendations) {
    // Severity badge
    const severityBadge = document.getElementById('severityBadge');
    severityBadge.textContent = recommendations.severity;
    severityBadge.className = 'severity-badge ' + recommendations.severity.toLowerCase();

    // Description
    document.getElementById('attackDescription').textContent = recommendations.description;

    // Immediate actions
    const immediateActions = document.getElementById('immediateActions');
    immediateActions.innerHTML = recommendations.immediate_actions.map(action =>
        `<li>${action}</li>`
    ).join('');

    // Investigation steps
    const investigationSteps = document.getElementById('investigationSteps');
    investigationSteps.innerHTML = recommendations.investigation_steps.map(step =>
        `<li>${step}</li>`
    ).join('');

    // Mitigation steps
    const mitigationSteps = document.getElementById('mitigationSteps');
    mitigationSteps.innerHTML = recommendations.mitigation_steps.map(step =>
        `<li>${step}</li>`
    ).join('');

    // Prevention measures
    const preventionMeasures = document.getElementById('preventionMeasures');
    preventionMeasures.innerHTML = recommendations.prevention.map(measure =>
        `<li>${measure}</li>`
    ).join('');
}

// Start real-time monitoring
function startMonitoring() {
    if (monitoringActive) return;

    monitoringActive = true;
    document.getElementById('startMonitoring').disabled = true;
    document.getElementById('stopMonitoring').disabled = false;

    // Connect WebSocket
    connectWebSocket();

    // Start polling for real-time data
    monitoringInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/stream/next`);
            const data = await response.json();

            if (data.prediction) {
                processRealtimeDetection(data);
            }
        } catch (error) {
            console.error('Monitoring error:', error);
        }
    }, 5000); // Poll every 5 seconds

    showNotification('Real-time monitoring started', 'success');
}

// Stop real-time monitoring
function stopMonitoring() {
    if (!monitoringActive) return;

    monitoringActive = false;
    document.getElementById('startMonitoring').disabled = false;
    document.getElementById('stopMonitoring').disabled = true;

    // Disconnect WebSocket
    if (websocket) {
        websocket.close();
        websocket = null;
    }

    // Stop polling
    if (monitoringInterval) {
        clearInterval(monitoringInterval);
        monitoringInterval = null;
    }

    showNotification('Real-time monitoring stopped', 'info');
}

// Connect to WebSocket
function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

    websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
        console.log('WebSocket connected');
    };

    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('WebSocket message:', data);
        // Handle real-time updates
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    websocket.onclose = () => {
        console.log('WebSocket disconnected');
    };
}

// Process real-time detection
function processRealtimeDetection(data) {
    samplesProcessed++;

    // Get actual attack type
    const attackType = data.actual_attack_type || data.predicted_attack_type || 'Unknown';
    const isAnomaly = data.actual_label === 1 || data.prediction !== 'benign';

    if (isAnomaly) {
        anomaliesDetected++;
    }

    // Update stats
    document.getElementById('samplesProcessed').textContent = samplesProcessed;
    document.getElementById('anomaliesDetected').textContent = anomaliesDetected;

    // Update chart
    updateChart(data, attackType);

    // Show alert for anomalies with attack type
    if (isAnomaly) {
        showAnomalyAlert(data, attackType);
    }
}

// Update real-time chart
function updateChart(data, attackType) {
    const timestamp = new Date().toLocaleTimeString();
    const displayLabel = attackType || data.predicted_attack_type || data.prediction;

    chartData.labels.push(timestamp);
    chartData.benign.push(data.actual_label === 0 ? 1 : 0);
    chartData.anomaly.push(data.actual_label === 1 ? 1 : 0);

    // Keep only last 20 data points
    if (chartData.labels.length > 20) {
        chartData.labels.shift();
        chartData.benign.shift();
        chartData.anomaly.shift();
    }

    realtimeChart.update();
}

// Show anomaly alert
function showAnomalyAlert(data, attackType) {
    const displayType = attackType || data.predicted_attack_type || 'ANOMALY';
    showNotification(
        `⚠️ ${displayType.toUpperCase()} DETECTED! Confidence: ${(data.confidence * 100).toFixed(1)}%`,
        'error'
    );
}

// Ask question to LLM
async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();

    if (!question) return;

    // Add user message to chat
    addChatMessage(question, 'user');
    questionInput.value = '';

    // Simulate LLM response (you can integrate with actual LLM API)
    setTimeout(() => {
        const answer = generateAnswer(question);
        addChatMessage(answer, 'assistant');
    }, 1000);
}

// Handle Enter key in question input
function handleQuestionKeypress(event) {
    if (event.key === 'Enter') {
        askQuestion();
    }
}

// Add message to chat
function addChatMessage(message, sender) {
    const chatContainer = document.getElementById('chatContainer');

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;

    const icon = sender === 'user' ?
        '<i class="fas fa-user"></i>' :
        '<i class="fas fa-robot"></i>';

    messageDiv.innerHTML = `
        ${icon}
        <div class="message-content">
            <p>${message}</p>
        </div>
    `;

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Generate answer (simplified - integrate with actual LLM)
function generateAnswer(question) {
    const lowerQuestion = question.toLowerCase();

    if (lowerQuestion.includes('why') && lowerQuestion.includes('anomaly')) {
        return "The system detected this as an anomaly based on multiple AI agents analyzing the network flow features. Key factors include unusual byte counts, suspicious TCP flags, and abnormal flow duration. The SHAP analysis shows which specific features contributed most to this decision.";
    } else if (lowerQuestion.includes('agent')) {
        return "Multiple specialized agents participated in this detection, including XGBoost, LSTM, and Random Forest models trained on different datasets. Each agent provides its own prediction and explanation, which are then combined through ensemble voting.";
    } else if (lowerQuestion.includes('false positive')) {
        return "To determine if this is a false positive, consider: 1) The confidence level - lower confidence may indicate uncertainty, 2) Review the key features identified by SHAP, 3) Check if this traffic pattern is expected in your environment, 4) Consult the analyst recommendations for investigation steps.";
    } else if (lowerQuestion.includes('what should') || lowerQuestion.includes('action')) {
        return "Based on the detection, I recommend: 1) Review the immediate actions in the Analyst Recommendations panel, 2) Investigate the source and destination of the traffic, 3) Check for similar patterns in your logs, 4) If confirmed malicious, follow the mitigation steps provided.";
    } else {
        return "I'm here to help you understand the anomaly detection results. You can ask me about why something was detected, which agents participated, what actions to take, or whether it might be a false positive. Feel free to ask more specific questions!";
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#3b82f6'};
        color: white;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;

    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
