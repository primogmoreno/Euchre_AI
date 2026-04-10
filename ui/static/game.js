/* ============================================================
   Euchre UI — game.js
   All frontend logic: state machine, rendering, API calls
   ============================================================ */

"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const SUIT_SYMBOL = { CLUBS: "♣", DIAMONDS: "♦", HEARTS: "♥", SPADES: "♠" };
const SUIT_COLOR  = { CLUBS: "black", DIAMONDS: "red", HEARTS: "red", SPADES: "black" };

// Which position maps to which grid slot from Player 0's perspective
// P0 = bottom, P1 = right, P2 = top, P3 = left
const POSITION_ZONE = { 1: "p1", 2: "p2", 3: "p3" };

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------
let gameState    = null;
let pendingAction = false;
let lastAgentConfig = null;   // saved so "Play Again" can re-use same agents

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  loadModels();
  document.getElementById("btn-start").addEventListener("click", onStartGame);
  document.getElementById("btn-new-game").addEventListener("click", showStartScreen);
  document.getElementById("btn-play-again").addEventListener("click", onPlayAgain);
  document.getElementById("btn-change-opponents").addEventListener("click", showStartScreen);
});

// ---------------------------------------------------------------------------
// Screen management
// ---------------------------------------------------------------------------
function showStartScreen() {
  document.getElementById("game-over-overlay").classList.add("hidden");
  document.getElementById("start-screen").classList.add("active");
  document.getElementById("game-screen").classList.remove("active");
  document.getElementById("start-error").classList.add("hidden");
}

function showGameScreen() {
  document.getElementById("start-screen").classList.remove("active");
  document.getElementById("game-screen").classList.add("active");
  document.getElementById("game-over-overlay").classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Load available models from server and populate selects
// ---------------------------------------------------------------------------
async function loadModels() {
  try {
    const res  = await fetch("/api/models");
    const data = await res.json();
    populateSelects(data);
  } catch (e) {
    console.error("Failed to load models:", e);
  }
}

function populateSelects(data) {
  const selIds = ["sel-p1", "sel-p2", "sel-p3"];
  selIds.forEach(id => {
    const sel = document.getElementById(id);
    sel.innerHTML = "";

    // Neural checkpoints group
    if (data.checkpoints && data.checkpoints.length > 0) {
      const grp = document.createElement("optgroup");
      grp.label = "Neural Network";
      data.checkpoints.forEach(ck => {
        const opt = document.createElement("option");
        opt.value = JSON.stringify({ type: "neural", checkpoint: ck.id });
        opt.textContent = ck.label;
        grp.appendChild(opt);
      });
      sel.appendChild(grp);
    }

    // Simple agents group
    if (data.agents && data.agents.length > 0) {
      const grp = document.createElement("optgroup");
      grp.label = "Simple Agents";
      data.agents.forEach(ag => {
        const opt = document.createElement("option");
        opt.value = JSON.stringify({ type: ag.id });
        opt.textContent = ag.label;
        grp.appendChild(opt);
      });
      sel.appendChild(grp);
    }
  });

  // Default selections: p1=final, p2=rule_based, p3=final
  setSelectDefault("sel-p1", "final_model");
  setSelectDefault("sel-p2", "rule_based");
  setSelectDefault("sel-p3", "final_model");
}

function setSelectDefault(selId, preferredId) {
  const sel = document.getElementById(selId);
  for (const opt of sel.options) {
    const cfg = JSON.parse(opt.value);
    if (cfg.checkpoint === preferredId || cfg.type === preferredId) {
      sel.value = opt.value;
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// Start game
// ---------------------------------------------------------------------------
async function onStartGame() {
  const errEl = document.getElementById("start-error");
  errEl.classList.add("hidden");

  const cfg = {
    player1: JSON.parse(document.getElementById("sel-p1").value),
    player2: JSON.parse(document.getElementById("sel-p2").value),
    player3: JSON.parse(document.getElementById("sel-p3").value),
  };
  lastAgentConfig = cfg;

  document.getElementById("btn-start").disabled = true;
  document.getElementById("btn-start").textContent = "Loading…";

  try {
    const res  = await fetch("/api/new_game", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(cfg),
    });
    const state = await res.json();
    if (!res.ok) {
      errEl.textContent = state.error || "Failed to start game.";
      errEl.classList.remove("hidden");
      return;
    }
    showGameScreen();
    render(state);
  } catch (e) {
    errEl.textContent = "Server error: " + e.message;
    errEl.classList.remove("hidden");
  } finally {
    document.getElementById("btn-start").disabled = false;
    document.getElementById("btn-start").textContent = "Start Game";
  }
}

async function onPlayAgain() {
  document.getElementById("game-over-overlay").classList.add("hidden");
  if (!lastAgentConfig) { showStartScreen(); return; }

  try {
    const res  = await fetch("/api/new_game", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(lastAgentConfig),
    });
    const state = await res.json();
    render(state);
  } catch (e) {
    console.error("Play again failed:", e);
    showStartScreen();
  }
}

// ---------------------------------------------------------------------------
// Send an action to the server
// ---------------------------------------------------------------------------
async function sendAction(payload) {
  if (pendingAction) return;
  pendingAction = true;

  try {
    const res   = await fetch("/api/action", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const state = await res.json();
    if (!res.ok) {
      console.warn("Illegal action or server error:", state.error);
      pendingAction = false;
      return;
    }
    render(state);
  } catch (e) {
    console.error("Action failed:", e);
  } finally {
    pendingAction = false;
  }
}

// ---------------------------------------------------------------------------
// Master render — called after every state update
// ---------------------------------------------------------------------------
function render(state) {
  gameState = state;

  renderScoreboard(state);
  renderOpponents(state);
  renderTurnedCard(state);
  renderCurrentTrick(state);
  renderPlayerHand(state);
  renderActionControls(state);
  renderEventLog(state);

  if (state.done) {
    renderGameOver(state);
  }
}

// ---------------------------------------------------------------------------
// Scoreboard
// ---------------------------------------------------------------------------
function renderScoreboard(state) {
  document.getElementById("score-0").textContent = state.score[0];
  document.getElementById("score-1").textContent = state.score[1];

  const trumpEl = document.getElementById("trump-display");
  if (state.trump) {
    const sym = SUIT_SYMBOL[state.trump];
    const col = SUIT_COLOR[state.trump];
    trumpEl.textContent  = "Trump: " + sym;
    trumpEl.style.color  = col === "red" ? "#ff6666" : "#e8c84a";
  } else {
    trumpEl.textContent = "";
  }

  const dealerEl = document.getElementById("dealer-display");
  const dealerNames = ["You", ...( state.player_names ? state.player_names.slice(1) : ["P1","P2","P3"])];
  dealerEl.textContent = "Dealer: " + (state.player_names ? state.player_names[state.dealer] : `P${state.dealer}`);

  // Tricks won badge (team 0 = you + P2)
  const tb = document.getElementById("tricks-badge-team0");
  if (tb) {
    const t0 = state.tricks_won[0];
    const t1 = state.tricks_won[1];
    if (t0 + t1 > 0) {
      tb.textContent = `Tricks: ${t0} – ${t1}`;
    } else {
      tb.textContent = "";
    }
  }
}

// ---------------------------------------------------------------------------
// Opponent zones (P1 right, P2 top, P3 left)
// ---------------------------------------------------------------------------
function renderOpponents(state) {
  const names     = state.player_names || ["You", "P1", "P2", "P3"];
  const handSizes = state.ai_hand_sizes || { 1: 5, 2: 5, 3: 5 };

  [1, 2, 3].forEach(p => {
    const zoneId = "zone-p" + p;
    const nameId = "name-p"  + p;
    const handId = "hand-p"  + p;

    const zone = document.getElementById(zoneId);
    const nameEl = document.getElementById(nameId);
    const handEl = document.getElementById(handId);

    if (!zone || !nameEl || !handEl) return;

    nameEl.textContent = names[p];

    // Sitting out
    const sittingOut = state.going_alone && state.caller !== null && ((state.caller + 2) % 4 === p);
    zone.classList.toggle("sitting-out", sittingOut);

    // Draw face-down cards
    const count = sittingOut ? 0 : (handSizes[p] ?? 5);
    handEl.innerHTML = "";
    for (let i = 0; i < count; i++) {
      const cb = document.createElement("div");
      cb.className = "card-back";
      handEl.appendChild(cb);
    }

    // Highlight if it's this player's turn
    zone.style.outline = (state.current_player === p && !state.done)
      ? "2px solid #e8c84a"
      : "none";
    zone.style.borderRadius = "8px";
  });
}

// ---------------------------------------------------------------------------
// Turned card (shown during calling phases)
// ---------------------------------------------------------------------------
function renderTurnedCard(state) {
  const area = document.getElementById("turned-card-area");
  area.innerHTML = "";

  if (state.turned_card && (state.phase === "CALLING_ROUND_1")) {
    const lbl = document.createElement("div");
    lbl.className = "turned-label";
    lbl.textContent = "Turned Up";
    area.appendChild(lbl);
    area.appendChild(buildCardEl(state.turned_card, false, false));
  }
}

// ---------------------------------------------------------------------------
// Current trick
// ---------------------------------------------------------------------------
function renderCurrentTrick(state) {
  const trickEl = document.getElementById("current-trick");
  trickEl.innerHTML = "";

  const names = state.player_names || ["You", "P1", "P2", "P3"];

  state.current_trick.forEach(entry => {
    const wrap = document.createElement("div");
    wrap.className = "trick-card-wrap";

    const playerLbl = document.createElement("div");
    playerLbl.className = "trick-player-label";
    playerLbl.textContent = names[entry.player];

    wrap.appendChild(playerLbl);
    wrap.appendChild(buildCardEl(entry.card, false, false));
    trickEl.appendChild(wrap);
  });
}

// ---------------------------------------------------------------------------
// Player hand (bottom — human)
// ---------------------------------------------------------------------------
function renderPlayerHand(state) {
  const handEl = document.getElementById("hand-p0");
  handEl.innerHTML = "";

  const legalCardIndices = new Set(
    state.legal_actions
      .filter(a => a.type === "card")
      .map(a => a.index)
  );

  const isCardPhase = state.phase === "PLAYING" || state.phase === "DISCARD";
  const isMyTurn    = state.current_player === 0 && !state.done;

  state.my_hand.forEach(card => {
    const isLegal = isCardPhase && isMyTurn && legalCardIndices.has(card.index);
    const el = buildCardEl(card, isLegal, false);

    if (isLegal) {
      el.addEventListener("click", () => {
        sendAction({ action: "card", card_index: card.index });
      });
    }

    handEl.appendChild(el);
  });
}

// ---------------------------------------------------------------------------
// Action controls (calling buttons / prompts)
// ---------------------------------------------------------------------------
function renderActionControls(state) {
  const area = document.getElementById("action-area");
  area.innerHTML = "";

  if (state.done) return;
  if (state.current_player !== 0) {
    const p = document.createElement("div");
    p.className = "action-prompt";
    p.textContent = "Waiting for AI…";
    area.appendChild(p);
    return;
  }

  const phase   = state.phase;
  const callers = state.legal_actions.filter(a => a.type === "call");

  if (phase === "CALLING_ROUND_1" || phase === "CALLING_ROUND_2") {
    const prompt = document.createElement("div");
    prompt.className = "action-prompt";
    prompt.textContent = phase === "CALLING_ROUND_1"
      ? "Order up or pass?"
      : "Name a suit or pass:";
    area.appendChild(prompt);

    callers.forEach(a => {
      const btn = document.createElement("button");
      btn.className = "btn-call";
      btn.textContent = a.display;

      if (a.action === "pass")                        btn.classList.add("pass");
      else if (a.action.includes("alone"))            btn.classList.add("alone");
      else if (a.action.startsWith("order_up"))       btn.classList.add("order");
      else                                            btn.classList.add("suit");

      btn.addEventListener("click", () => sendAction({ action: a.action }));
      area.appendChild(btn);
    });

  } else if (phase === "DISCARD") {
    const prompt = document.createElement("div");
    prompt.className = "action-prompt";
    prompt.textContent = "Click a card to discard";
    area.appendChild(prompt);
  }
  // PLAYING: card clicks handled in renderPlayerHand — no extra buttons needed
}

// ---------------------------------------------------------------------------
// Event log
// ---------------------------------------------------------------------------
function renderEventLog(state) {
  const logEl = document.getElementById("event-log");
  logEl.innerHTML = "";

  const events = state.events || [];
  events.forEach(msg => {
    const div = document.createElement("div");
    div.className = "event-entry" + (msg.startsWith("---") ? " round-event" : "");
    div.textContent = msg;
    logEl.appendChild(div);
  });

  // Auto-scroll to bottom
  logEl.scrollTop = logEl.scrollHeight;
}

// ---------------------------------------------------------------------------
// Game over
// ---------------------------------------------------------------------------
function renderGameOver(state) {
  const overlay = document.getElementById("game-over-overlay");
  const titleEl = document.getElementById("game-over-title");
  const scoreEl = document.getElementById("game-over-score");

  const youWon = state.winner === "Your team";
  titleEl.textContent = youWon ? "You Win!" : "You Lose";
  titleEl.className   = "game-over-title " + (youWon ? "win" : "lose");
  scoreEl.textContent = `Final Score: Your Team ${state.score[0]} – Opponents ${state.score[1]}`;

  overlay.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Card element builder
// ---------------------------------------------------------------------------
function buildCardEl(card, isLegal, isSelected) {
  const el = document.createElement("div");
  el.className = "card";
  el.classList.add(card.color);                        // "red" or "black"
  if (isLegal)    el.classList.add("legal");
  if (isSelected) el.classList.add("selected");
  if (!isLegal && !isSelected) el.classList.add("illegal");

  const sym = SUIT_SYMBOL[card.suit] || card.suit;

  const top = document.createElement("span");
  top.className   = "rank-top";
  top.textContent = card.rank;

  const center = document.createElement("span");
  center.className   = "suit-center";
  center.textContent = sym;

  const bot = document.createElement("span");
  bot.className   = "rank-bot";
  bot.textContent = card.rank;

  el.appendChild(top);
  el.appendChild(center);
  el.appendChild(bot);

  el.dataset.index = card.index;
  return el;
}
