// Popup script for Bangla SafeNet Chrome Extension

document.addEventListener("DOMContentLoaded", () => {
  const commentInput = document.getElementById("commentInput");
  const thresholdRange = document.getElementById("thresholdRange");
  const thresholdVal = document.getElementById("thresholdVal");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const loader = document.getElementById("loader");
  const resultCard = document.getElementById("resultCard");
  const badge = document.getElementById("badge");
  const scoreText = document.getElementById("scoreText");
  const meterFill = document.getElementById("meterFill");
  const flaggedSection = document.getElementById("flaggedSection");
  const flaggedWords = document.getElementById("flaggedWords");
  const apiUrlInput = document.getElementById("apiUrl");
  const saveApiBtn = document.getElementById("saveApiBtn");

  // Load saved API endpoint
  if (chrome && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["banglaSafeNetApi"], (result) => {
      if (result.banglaSafeNetApi) {
        apiUrlInput.value = result.banglaSafeNetApi;
      }
    });
  }

  saveApiBtn.addEventListener("click", () => {
    const url = apiUrlInput.value.trim() || "http://localhost:8000";
    if (chrome && chrome.storage && chrome.storage.local) {
      chrome.storage.local.set({ banglaSafeNetApi: url }, () => {
        alert("API URL সংরক্ষিত হয়েছে!");
      });
    } else {
      localStorage.setItem("banglaSafeNetApi", url);
      alert("API URL সংরক্ষিত হয়েছে!");
    }
  });

  thresholdRange.addEventListener("input", (e) => {
    thresholdVal.textContent = `${e.target.value}%`;
  });

  analyzeBtn.addEventListener("click", async () => {
    const text = commentInput.value.trim();
    if (!text) {
      alert("অনুগ্রহ করে একটি মন্তব্য লিখুন (Please enter a comment)!");
      return;
    }

    const threshold = parseFloat(thresholdRange.value) / 100.0;
    const baseApi = apiUrlInput.value.trim().replace(/\/+$/, "");

    loader.classList.remove("hidden");
    resultCard.classList.add("hidden");

    try {
      const response = await fetch(`${baseApi}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, threshold }),
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      displayResult(data);
    } catch (err) {
      console.warn("Backend unavailable, using local client heuristics:", err);
      // Fallback local heuristic inspection
      const localResult = localAnalyze(text, threshold);
      displayResult(localResult);
    } finally {
      loader.classList.add("hidden");
    }
  });

  function displayResult(data) {
    resultCard.classList.remove("hidden");

    const pct = Math.round(data.toxicity_score * 100);
    scoreText.textContent = `টক্সিক স্কোর: ${pct}%`;
    meterFill.style.width = `${pct}%`;

    if (data.is_toxic) {
      badge.textContent = "🚨 ক্ষতিকর / আপত্তিকর";
      badge.className = "badge toxic";
      meterFill.style.backgroundColor = "#ef4444";
    } else {
      badge.textContent = "✅ নিরাপদ মন্তব্য";
      badge.className = "badge safe";
      meterFill.style.backgroundColor = "#10b981";
    }

    if (data.flagged_spans && data.flagged_spans.length > 0) {
      flaggedWords.textContent = data.flagged_spans.join(", ");
      flaggedSection.classList.remove("hidden");
    } else {
      flaggedSection.classList.add("hidden");
    }
  }

  function localAnalyze(text, threshold) {
    const toxicList = ["শালা", "শালি", "মাগি", "বেশ্যা", "খানকি", "শুয়োরের", "কুত্তা", "হারামি", "বদমাইশ", "মর", "মারব", "ফালতু"];
    const found = toxicList.filter((w) => text.includes(w));
    const isToxic = found.length > 0;
    const score = isToxic ? 0.85 : 0.12;

    return {
      original_text: text,
      is_toxic: score >= threshold,
      label: score >= threshold ? "Toxic / Harassment" : "Non-Toxic",
      toxicity_score: score,
      flagged_spans: found,
    };
  }
});
