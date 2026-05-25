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

let isLoading = false;

async function sendMessage() {
    const query = queryInput.value.trim();
    if (!query || isLoading) return;

    if (welcomeContainer) {
        welcomeContainer.style.display = "none";
    }

    addMessage(query, "user");
    queryInput.value = "";
    queryInput.style.height = "auto";
    setLoading(true);
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
        hideTypingIndicator();
        addAIMessage(data.answer);
        updateSourcesPanel(data.sources || []);
        updateConfidenceScore(data.confidence || 0);

        if (data.hallucination_flags && data.hallucination_flags.length > 0) {
            showHallucinationWarning(data.hallucination_flags);
        } else {
            hideHallucinationWarning();
        }

        updateTimingInfo(data.timing || {});
    } catch (error) {
        hideTypingIndicator();
        addAIMessage(`Error: ${error.message}\n\nMake sure the RAG pipeline is initialized. Run ingest_documents.py first.`);
        console.error("Query error:", error);
    } finally {
        setLoading(false);
    }
}

function addMessage(text, role) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", `message-${role}`);

    const avatar = document.createElement("div");
    avatar.classList.add(role === "user" ? "user-avatar" : "ai-avatar");
    avatar.textContent = role === "user" ? "You" : "SG";

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
    scrollToBottom();
}

function addAIMessage(text) {
    addMessage(text, "ai");
}

function showTypingIndicator() {
    typingIndicator.style.display = "flex";
    scrollToBottom();
}

function hideTypingIndicator() {
    typingIndicator.style.display = "none";
}

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
                <span class="source-score">${scorePercent}%</span>
            </div>
            <div class="source-article">Art. ${escapeHtml(source.article_number || "N/A")}</div>
            <div class="source-text">${escapeHtml(previewText)}</div>
        `;

        sourceCards.appendChild(card);
    });

    sourcesSidebar.classList.remove("collapsed");
}

function updateConfidenceScore(confidence) {
    const percent = Math.round(confidence * 100);
    confidenceSection.style.display = "block";

    setTimeout(() => {
        confidenceFill.style.width = `${percent}%`;
    }, 100);

    confidenceValue.textContent = `${percent}%`;

    if (percent >= 70) {
        confidenceValue.style.color = "#10b981";
    } else if (percent >= 40) {
        confidenceValue.style.color = "#c8a44e";
    } else {
        confidenceValue.style.color = "#ef4444";
    }
}

function showHallucinationWarning(flags) {
    hallucinationWarning.style.display = "flex";
    warningText.textContent = `${flags.length} potential issue(s): ${flags.join(", ")}`;
}

function hideHallucinationWarning() {
    hallucinationWarning.style.display = "none";
}

function updateTimingInfo(timing) {
    timingInfo.style.display = "block";
    document.getElementById("timing-retrieval").textContent = `${(timing.retrieval || 0).toFixed(3)}s`;
    document.getElementById("timing-generation").textContent = `${(timing.generation || 0).toFixed(3)}s`;
    document.getElementById("timing-total").textContent = `${(timing.total || 0).toFixed(3)}s`;
}

function handleSuggestionClick(query) {
    queryInput.value = query;
    sendMessage();
}

function toggleSidebar() {
    sourcesSidebar.classList.toggle("collapsed");
}

function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
});

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

function formatMarkdown(text) {
    if (!text) return "";
    return text
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text || "";
    return div.innerHTML;
}

queryInput.focus();
