/**
 * Better Call Saul AI — Frontend JavaScript
 * ==========================================
 *
 * Handles all chat interactivity:
 * - Sending messages to the Flask API
 * - Rendering user and AI messages
 * - Typing indicator animation
 * - Sources panel updates
 * - Confidence meter
 * - Hallucination warnings
 * - Suggestion chips
 * - Keyboard shortcuts (Enter to send, Shift+Enter for newline)
 */

// =============================================================================
// DOM References
// =============================================================================

const queryInput = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const messagesContainer = document.getElementById("messages-container");
const welcomeContainer = document.getElementById("welcome-container");
const typingIndicator = document.getElementById("typing-indicator");
const sourcesSidebar = document.getElementById("sources-sidebar");
const sourceCards = document.getElementById("source-cards");
const confidenceSection = document.getElementById("confidence-section");
const confidenceFill = document.getElementById("confidence-fill");
const confidenceValue = document.getElementById("confidence-value");
const hallucinationWarning = document.getElementById("hallucination-warning");
const warningText = document.getElementById("warning-text");
const timingInfo = document.getElementById("timing-info");

// State
let isLoading = false;

// =============================================================================
// Message Handling
// =============================================================================

/**
 * Send the user's message to the API and display the response.
 */
async function sendMessage() {
    const query = queryInput.value.trim();
    if (!query || isLoading) return;

    // Hide welcome screen on first message
    if (welcomeContainer) {
        welcomeContainer.style.display = "none";
    }

    // Add user message to chat
    addMessage(query, "user");

    // Clear input and reset height
    queryInput.value = "";
    queryInput.style.height = "auto";

    // Disable input while loading
    setLoading(true);

    // Show typing indicator
    showTypingIndicator();

    try {
        const response = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const data = await response.json();

        // Hide typing indicator
        hideTypingIndicator();

        // Add AI message
        addAIMessage(data.answer);

        // Update sources panel
        updateSourcesPanel(data.sources || []);

        // Update confidence score
        updateConfidenceScore(data.confidence || 0);

        // Update hallucination warnings
        if (data.hallucination_flags && data.hallucination_flags.length > 0) {
            showHallucinationWarning(data.hallucination_flags);
        } else {
            hideHallucinationWarning();
        }

        // Update timing info
        updateTimingInfo(data.timing || {});

    } catch (error) {
        hideTypingIndicator();
        addAIMessage(`❌ Sorry, I encountered an error: ${error.message}\n\nMake sure the RAG pipeline is initialized — run \`ingest_documents.py\` first.`);
        console.error("Query error:", error);
    } finally {
        setLoading(false);
    }
}

/**
 * Add a message bubble to the chat.
 * @param {string} text - The message text
 * @param {string} role - "user" or "ai"
 */
function addMessage(text, role) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", `message-${role}`);

    const avatar = document.createElement("div");
    avatar.classList.add(role === "user" ? "user-avatar" : "ai-avatar");
    avatar.textContent = role === "user" ? "You" : "⚖️";

    const bubble = document.createElement("div");
    bubble.classList.add("message-bubble");

    if (role === "user") {
        bubble.textContent = text;
    } else {
        bubble.innerHTML = formatMarkdown(text);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(bubble);
    messagesContainer.appendChild(messageDiv);

    // Auto-scroll to bottom
    scrollToBottom();
}

/**
 * Add an AI message (convenience wrapper for formatted responses).
 * @param {string} text - The AI response text
 */
function addAIMessage(text) {
    addMessage(text, "ai");
}

// =============================================================================
// Typing Indicator
// =============================================================================

function showTypingIndicator() {
    typingIndicator.style.display = "flex";
    scrollToBottom();
}

function hideTypingIndicator() {
    typingIndicator.style.display = "none";
}

// =============================================================================
// Sources Panel
// =============================================================================

/**
 * Update the sources sidebar with retrieved documents.
 * @param {Array} sources - Array of source objects from the API
 */
function updateSourcesPanel(sources) {
    sourceCards.innerHTML = "";

    if (!sources || sources.length === 0) {
        sourceCards.innerHTML = '<p class="no-sources">No sources retrieved.</p>';
        return;
    }

    sources.forEach((source, index) => {
        const card = document.createElement("div");
        card.classList.add("source-card");
        card.onclick = () => card.classList.toggle("expanded");

        const scorePercent = Math.round((source.score || 0) * 100);
        const previewText = source.text ? source.text.substring(0, 200) + (source.text.length > 200 ? "..." : "") : "No text available";

        card.innerHTML = `
            <div class="source-card-header">
                <span class="source-law">${escapeHtml(source.law_name || "Unknown Law")}</span>
                <span class="source-score">${scorePercent}% match</span>
            </div>
            <div class="source-article">📌 Article ${escapeHtml(source.article_number || "N/A")}</div>
            <div class="source-text">${escapeHtml(previewText)}</div>
        `;

        sourceCards.appendChild(card);
    });

    // Ensure sidebar is visible
    sourcesSidebar.classList.remove("collapsed");
}

// =============================================================================
// Confidence Score
// =============================================================================

/**
 * Update the confidence meter in the sidebar.
 * @param {number} confidence - Confidence score (0 to 1)
 */
function updateConfidenceScore(confidence) {
    const percent = Math.round(confidence * 100);
    confidenceSection.style.display = "block";

    // Animate the fill
    setTimeout(() => {
        confidenceFill.style.width = `${percent}%`;
    }, 100);

    confidenceValue.textContent = `${percent}%`;

    // Color code
    if (percent >= 70) {
        confidenceValue.style.color = "#10b981"; // green
    } else if (percent >= 40) {
        confidenceValue.style.color = "#f59e0b"; // amber
    } else {
        confidenceValue.style.color = "#ef4444"; // red
    }
}

// =============================================================================
// Hallucination Warning
// =============================================================================

function showHallucinationWarning(flags) {
    hallucinationWarning.style.display = "flex";
    warningText.textContent = `⚠️ ${flags.length} potential issue(s) detected: ${flags.join(", ")}`;
}

function hideHallucinationWarning() {
    hallucinationWarning.style.display = "none";
}

// =============================================================================
// Timing Info
// =============================================================================

function updateTimingInfo(timing) {
    timingInfo.style.display = "block";
    document.getElementById("timing-retrieval").textContent = `${(timing.retrieval || 0).toFixed(3)}s`;
    document.getElementById("timing-generation").textContent = `${(timing.generation || 0).toFixed(3)}s`;
    document.getElementById("timing-total").textContent = `${(timing.total || 0).toFixed(3)}s`;
}

// =============================================================================
// Suggestion Chips
// =============================================================================

/**
 * Handle clicking a suggestion chip.
 * @param {string} query - The suggested query text
 */
function handleSuggestionClick(query) {
    queryInput.value = query;
    sendMessage();
}

// =============================================================================
// Sidebar Toggle
// =============================================================================

function toggleSidebar() {
    sourcesSidebar.classList.toggle("collapsed");
}

// =============================================================================
// Input Handling
// =============================================================================

/**
 * Handle keyboard events in the input field.
 * Enter = send, Shift+Enter = newline
 */
function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

/**
 * Auto-resize the textarea as the user types.
 */
queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
});

// =============================================================================
// Utility Functions
// =============================================================================

function setLoading(loading) {
    isLoading = loading;
    sendBtn.disabled = loading;
    queryInput.disabled = loading;
}

function scrollToBottom() {
    const chatArea = document.querySelector(".chat-area");
    setTimeout(() => {
        chatArea.scrollTop = chatArea.scrollHeight;
    }, 50);
}

/**
 * Basic markdown formatting for AI responses.
 * Handles bold, italic, code, and line breaks.
 */
function formatMarkdown(text) {
    if (!text) return "";

    return text
        // Code blocks (must come before inline code)
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Bold
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // Line breaks
        .replace(/\n/g, '<br>');
}

/**
 * Escape HTML to prevent XSS.
 */
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text || "";
    return div.innerHTML;
}

// =============================================================================
// Initialization
// =============================================================================

// Focus the input on page load
queryInput.focus();

console.log("⚖️ Better Call Saul AI — Ready!");
console.log("   In legal trouble? Better Call Saul AI!");
