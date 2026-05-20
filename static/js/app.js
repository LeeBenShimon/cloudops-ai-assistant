document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("chat-form");
  const textarea = document.getElementById("message-input");
  const messages = document.getElementById("messages");
  const resetButton = document.getElementById("reset-session");
  const sessionIdNode = document.getElementById("session-id");
  const messageCountNode = document.getElementById("message-count");
  const quickPrompts = document.querySelectorAll("[data-prompt]");

  const state = {
    busy: false,
  };

  function scrollMessagesToEnd() {
    if (messages) {
      messages.scrollTop = messages.scrollHeight;
    }
  }

  function createMessage(role, content, createdAt = new Date()) {
    const article = document.createElement("article");
    article.className = `message message--${role}`;

    const meta = document.createElement("div");
    meta.className = "message__meta";

    const roleLabel = document.createElement("span");
    roleLabel.textContent = role;

    const time = document.createElement("time");
    time.textContent = createdAt.toLocaleString([], {
      hour: "2-digit",
      minute: "2-digit",
      day: "2-digit",
      month: "short",
    });

    const body = document.createElement("div");
    body.className = "message__body";
    body.textContent = content;

    meta.append(roleLabel, time);
    article.append(meta, body);
    return article;
  }

  function ensureEmptyStateRemoved() {
    const emptyState = messages?.querySelector(".empty-state");
    if (emptyState) {
      emptyState.remove();
    }
  }

  function setBusy(flag) {
    state.busy = flag;
    textarea.disabled = flag;
    form.querySelector('button[type="submit"]').disabled = flag;
    if (resetButton) {
      resetButton.disabled = flag;
    }
  }

  async function submitMessage(text) {
    ensureEmptyStateRemoved();
    messages.appendChild(createMessage("user", text));

    const loadingSteps = [
  "Searching knowledge base...",
  "Retrieving relevant chunks...",
  "Generating grounded response..."
];

let loadingIndex = 0;

const typing = createMessage(
  "assistant",
  loadingSteps[0]
);

typing.classList.add("message--typing");

const loadingInterval = setInterval(() => {

  loadingIndex =
    (loadingIndex + 1) % loadingSteps.length;

  const body = typing.querySelector(".message__body");

  if (body) {
    body.textContent = loadingSteps[loadingIndex];
  }

}, 1200);

messages.appendChild(typing);

scrollMessagesToEnd();
    typing.classList.add("message--typing");
    messages.appendChild(typing);
    scrollMessagesToEnd();

    setBusy(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: text }),
      });

      const data = await response.json();

      clearInterval(loadingInterval);

      typing.remove();

      if (!response.ok) {
        throw new Error(
          data.error || "The assistant could not answer right now."
        );
      }

      messages.appendChild(
        createMessage("assistant", data.answer)
      );

      renderSources(data.retrieved);

      if (sessionIdNode && data.session_id) {
        sessionIdNode.textContent =
          `${data.session_id.slice(0, 8)}…`;
      }

      if (messageCountNode) {
        const currentCount = Number(messageCountNode.textContent || "0");
        messageCountNode.textContent = String(currentCount + 2);
      }

      document.body.dataset.sessionId = data.session_id || document.body.dataset.sessionId;
      document.body.dataset.sessionTitle = data.session_title || document.body.dataset.sessionTitle;
      scrollMessagesToEnd();
    } catch (error) {
      clearInterval(loadingInterval);
      typing.remove();
      messages.appendChild(createMessage("assistant", error.message));
      scrollMessagesToEnd();
    } finally {
      setBusy(false);
      textarea.focus();
    }
  }

  if (form && textarea) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (state.busy) {
        return;
      }

      const text = textarea.value.trim();
      if (!text) {
        textarea.focus();
        return;
      }

      textarea.value = "";
      await submitMessage(text);
    });

    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && event.shiftKey) {
        return;
      }

      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
  }

  quickPrompts.forEach((button) => {
    button.addEventListener("click", () => {
      textarea.value = button.dataset.prompt || "";
      textarea.focus();
      textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    });
  });

  if (resetButton) {
    resetButton.addEventListener("click", async () => {
      if (state.busy) {
        return;
      }

      setBusy(true);

      try {
        const response = await fetch("/api/reset", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Unable to reset the session.");
        }

        messages.innerHTML = `
          <div class="empty-state">
            <h3>Start the conversation</h3>
            <p>Use a prompt above or write your own question. The assistant will remember the chat in SQLite for this browser session.</p>
          </div>
        `;

        textarea.value = "";
        if (sessionIdNode) {
          sessionIdNode.textContent = `${data.session_id.slice(0, 8)}…`;
        }
        if (messageCountNode) {
          messageCountNode.textContent = "0";
        }
        document.body.dataset.sessionId = data.session_id;
        document.body.dataset.sessionTitle = data.session_title;
        textarea.focus();
      } catch (error) {
        window.alert(error.message);
      } finally {
        setBusy(false);
      }
    });
  }

  scrollMessagesToEnd();
  const uploadZone = document.getElementById("upload-zone");
  const uploadInput = document.getElementById("file-upload");

const uploadStatus = document.getElementById("upload-status");

if (uploadZone) {

    ["dragenter", "dragover"].forEach((eventName) => {

        uploadZone.addEventListener(eventName, (event) => {

            event.preventDefault();

            uploadZone.classList.add("upload-zone--active");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {

        uploadZone.addEventListener(eventName, () => {

            uploadZone.classList.remove("upload-zone--active");
        });
    });

    uploadZone.addEventListener("drop", async (event) => {

        event.preventDefault();

        const file = event.dataTransfer.files[0];

        if (!file) return;

        uploadInput.files = event.dataTransfer.files;

        uploadInput.dispatchEvent(
            new Event("change")
        );
    });
}

if (uploadInput) {

    uploadInput.addEventListener("change", async (event) => {

        const file = event.target.files[0];

        if (!file) return;

        uploadStatus.innerHTML = "Uploading file...";

        const formData = new FormData();

        formData.append("file", file);

        try {

            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (data.success) {

                uploadStatus.innerHTML = `
                    ✅ Successfully indexed:
                    <strong>${data.filename}</strong>
                `;

                loadUploadedFiles();

                setTimeout(() => {

                    uploadStatus.innerHTML = "";

                }, 4000);

            } else {

                uploadStatus.innerHTML =
                    `❌ ${data.error}`;
            }


        } catch (error) {

            uploadStatus.innerHTML =
                "❌ Upload failed.";
        }
    });
}

function renderSources(sources) {

    const panel = document.getElementById("sources-panel");

    const list = document.getElementById("sources-list");

    if (!sources || sources.length === 0) {

        panel.classList.add("hidden");

        return;
    }

    panel.classList.remove("hidden");

    list.innerHTML = "";

    sources.forEach((source) => {

        const item = document.createElement("div");

        item.className = "source-item";

        item.innerHTML = `
            <div class="source-file">
                📄 ${source.source}
            </div>

            <div class="source-score">
                Similarity: ${source.score.toFixed(2)}
            </div>
        `;

        list.appendChild(item);
    });
}

async function loadUploadedFiles() {

    const list =
      document.getElementById(
        "uploaded-files-list"
      );

    if (!list) return;

    try {

        const response =
          await fetch("/api/files");

        const data =
          await response.json();

        list.innerHTML = "";

        if (data.files.length === 0) {

          list.innerHTML = `
              <div class="uploaded-file empty-upload">

                  No documents uploaded yet

              </div>
          `;

    return;
}

        data.files.forEach((file) => {

            const item =
              document.createElement("div");

            item.className = "uploaded-file";

            item.innerHTML = `
                📄 ${file}
            `;

            list.appendChild(item);
        });

    } catch (error) {

        console.error(
          "Failed to load files",
          error
        );
    }
}

loadUploadedFiles();

});