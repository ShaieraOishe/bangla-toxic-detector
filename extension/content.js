// Content script for Bangla SafeNet
// Can monitor comment inputs and alert if potentially toxic content is typed

console.log("🛡️ Bangla SafeNet Content Script loaded.");

const TOXIC_PATTERNS = [
  "শালা", "শালি", "মাগি", "বেশ্যা", "খানকি", "শুয়োরের", "কুত্তা",
  "হারামি", "বদমাইশ", "মর", "মারব", "ফালতু"
];

function checkText(text) {
  return TOXIC_PATTERNS.filter(word => text.includes(word));
}

// Listen for typing events on social media comment boxes
document.addEventListener("input", (e) => {
  const target = e.target;
  if (target && (target.tagName === "TEXTAREA" || target.getAttribute("contenteditable") === "true")) {
    const text = target.value || target.innerText || "";
    const flagged = checkText(text);

    if (flagged.length > 0) {
      target.style.borderColor = "#ef4444";
      target.title = `⚠️ সাবধান: আপত্তিকর শব্দ সনাক্ত হয়েছে (${flagged.join(", ")})`;
    } else {
      target.style.borderColor = "";
      target.title = "";
    }
  }
}, true);
