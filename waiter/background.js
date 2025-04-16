let currentlyDelayedUrls = new Set();

chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  if (details.url.startsWith(chrome.runtime.getURL(""))) {
    return;
  }

  // Skip iframes and subframes - only process main frame navigations
  if (details.frameId !== 0) {
    return;
  }

  if (currentlyDelayedUrls.has(details.url)) {
    currentlyDelayedUrls.delete(details.url);
    return;
  }

  const websites = await new Promise(resolve => {
    chrome.storage.sync.get(['websites'], (result) => {
      resolve(result.websites || []);
    });
  });
  
  const hostname = new URL(details.url).hostname.replace('www.', '');
  
  if (websites.some(site => hostname.includes(site.replace('www.', '')))) {
    chrome.storage.local.set({ delayedUrl: details.url });
    currentlyDelayedUrls.add(details.url);
    chrome.tabs.update(details.tabId, {
      url: chrome.runtime.getURL("waiting.html")
    });
  }
});
