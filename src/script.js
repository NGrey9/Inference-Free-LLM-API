const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const fileInput = document.getElementById("file-input");
const modelSelect = document.getElementById("model-select");

// Replace with your backend API endpoint
const API_URL = "http://localhost:3000/chat";
const MODEL_API_URL = "http://localhost:3000/model";

let selectedFile = null;


function appendMessage(sender, text) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;
    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "bubble";
    bubbleDiv.textContent = text;
    messageDiv.appendChild(bubbleDiv);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight; 
}

fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file; 
        appendMessage("user", `Selected file: ${file.name}`);
    }
});


async function sendMessage() {
    const userMessage = userInput.value.trim();
    
    if (userMessage === "") return; 

    appendMessage("user", userMessage); 
    userInput.value = ""; 

    const formData = new FormData();
    formData.append("message", userMessage);

    if (selectedFile) {
        formData.append("file", selectedFile); 
    }

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const data = await response.json();
            appendMessage("bot", data.processed_message); 
        } else {
            appendMessage("bot", "Error: Unable to process the message.");
        }
    } catch (error) {
        appendMessage("bot", "Error: Unable to connect to the server.");
    }
}



async function changeModel(selectedModel) {
    console.log("Change model initiated with:", selectedModel); 

    try {
        const response = await fetch(MODEL_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_name: selectedModel }), 
        });

        console.log("Response status:", response.status); 

        if (response.ok) {
            const data = await response.json();
            console.log("Response data:", data); 
            appendMessage("bot", `Model switched to: ${data.model_name || selectedModel}`);
        } else {
            const errorText = await response.text(); 
            console.error("Error response text:", errorText); 
            appendMessage("bot", "Error: Unable to switch the model.");
        }
    } catch (error) {
        console.error("Fetch error:", error); 
        appendMessage("bot", "Error: Unable to connect to the server for model switching.");
    }
}


// Event listeners
sendButton.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});


modelSelect.addEventListener("change", (e) => {
    const selectedModel = e.target.value; 
    appendMessage("bot", `Changing model to: ${selectedModel}`);
    changeModel(selectedModel); 
});
