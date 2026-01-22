const dashboard = {
  statusPill: document.getElementById("status-pill"),
  projectName: document.getElementById("project-name"),
  sprintProgress: document.getElementById("sprint-progress"),
  sprintEta: document.getElementById("sprint-eta"),
  activeAgent: document.getElementById("active-agent"),
  progressBar: document.getElementById("progress-bar"),
  storyCount: document.getElementById("story-count"),
  gateList: document.getElementById("gate-list"),
  agentGrid: document.getElementById("agent-grid"),
  storyList: document.getElementById("story-list"),
  auditList: document.getElementById("audit-list"),
};

function updateDashboard(data) {
  dashboard.projectName.textContent = data.sprint.project_name || "Project";
  dashboard.sprintProgress.textContent = `${Math.round(data.sprint.progress * 100)}%`;
  dashboard.sprintEta.textContent = `${data.sprint.eta_minutes} min`;
  dashboard.activeAgent.textContent = data.sprint.active_agent;
  dashboard.progressBar.style.width = `${Math.round(data.sprint.progress * 100)}%`;
  dashboard.storyCount.textContent = `${data.sprint.stories_completed} of ${data.sprint.stories_total} stories`;
  dashboard.statusPill.textContent = data.sprint.active_agent
    ? `Active: ${data.sprint.active_agent}`
    : "Idle";

  dashboard.gateList.innerHTML = "";
  data.gates.forEach((gate) => {
    const item = document.createElement("div");
    item.className = "gate-item";
    item.innerHTML = `<span>${gate.name}</span><span>${gate.score}</span>`;
    dashboard.gateList.appendChild(item);
  });

  dashboard.agentGrid.innerHTML = "";
  data.agents.forEach((agent) => {
    const card = document.createElement("div");
    card.className = `agent-card ${agent.state}`;
    card.textContent = `${agent.name}\n${agent.state}`;
    dashboard.agentGrid.appendChild(card);
  });

  dashboard.storyList.innerHTML = "";
  data.stories.forEach((story) => {
    const item = document.createElement("li");
    item.className = "story-item";
    item.textContent = `${story.id}: ${story.status}`;
    dashboard.storyList.appendChild(item);
  });

  dashboard.auditList.innerHTML = "";
  data.audit.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "audit-item";
    item.innerHTML = `<time>${entry.timestamp}</time>${entry.agent}: ${entry.decision}`;
    dashboard.auditList.appendChild(item);
  });
}

async function fetchDashboard() {
  const response = await fetch("/api/v1/dashboard");
  const data = await response.json();
  updateDashboard(data);
}

function connectWebSocket() {
  const socket = new WebSocket(`ws://${window.location.host}/ws`);
  socket.onmessage = () => {
    fetchDashboard();
  };
  socket.onclose = () => {
    setTimeout(connectWebSocket, 3000);
  };
}

fetchDashboard();
connectWebSocket();
