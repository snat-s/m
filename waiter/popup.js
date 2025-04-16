document.addEventListener('DOMContentLoaded', initPopup);

async function initPopup() {
  if (window.location.pathname.endsWith('/waiting.html')) {
    initWaitingPage();
  } else {
    initPopupPage();
  }
}

async function initPopupPage() {
  console.log("Popup initialized");
  const websiteInput = document.getElementById('website-input');
  const addBtn = document.getElementById('add-btn');
  const websiteList = document.getElementById('website-list');
  const delayTimeInput = document.getElementById('delay-time');
  const saveSettingsBtn = document.getElementById('save-settings-btn');

  let websites = await loadWebsites();
  let delayTime = await loadDelayTime();
  renderWebsiteList(websiteList, websites);

  delayTimeInput.value = delayTime;

  addBtn.addEventListener('click', () => {
    console.log("Add button clicked");
    const url = websiteInput.value.trim();
    if (url && !websites.includes(url)) {
      websites.push(url);
      saveWebsites(websites);
      renderWebsiteList(websiteList, websites);
      websiteInput.value = '';
      console.log("Added website:", url);
    } else {
      console.log("Website not added, empty or duplicate:", url);
    }
  });

  saveSettingsBtn.addEventListener('click', () => {
    const newDelayTime = parseInt(delayTimeInput.value, 10) || 5;
    const validatedTime = Math.min(60, Math.max(1, newDelayTime));
    delayTimeInput.value = validatedTime;
    saveDelayTime(validatedTime);
  });
}

async function initWaitingPage() {
  const delayTime = await loadDelayTime();
  let seconds = delayTime;
  const countdown = document.getElementById('countdown');
  countdown.textContent = `[${seconds}]`;
  
  const timer = setInterval(() => {
    seconds--;
    countdown.textContent = `[${seconds}]`;
    if (seconds <= 0) {
      clearInterval(timer);
      chrome.storage.local.get(['delayedUrl'], (result) => {
        if (result.delayedUrl) {
          window.location.href = result.delayedUrl;
        }
      });
    }
  }, 1000);
}

function renderWebsiteList(container, websites) {
  container.innerHTML = '';
  websites.forEach(url => {
    const item = document.createElement('div');
    item.className = 'website-item';
    const urlSpan = document.createElement('span');
    urlSpan.textContent = url;
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '[-]';
    deleteBtn.className = 'delete-btn';
    deleteBtn.addEventListener('click', () => {
      websites = websites.filter(u => u !== url);
      saveWebsites(websites);
      renderWebsiteList(container, websites);
    });
    item.appendChild(urlSpan);
    item.appendChild(deleteBtn);
    container.appendChild(item);
  });
}

async function loadWebsites() {
  return new Promise(resolve => {
    chrome.storage.sync.get(['websites'], (result) => {
      resolve(result.websites || []);
    });
  });
}

function saveWebsites(websites) {
  chrome.storage.sync.set({ websites });
}

async function loadDelayTime() {
  return new Promise(resolve => {
    chrome.storage.sync.get(['delayTime'], (result) => {
      resolve(result.delayTime || 5);
    });
  });
}

function saveDelayTime(delayTime) {
  chrome.storage.sync.set({ delayTime });
}
