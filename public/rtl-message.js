/*
 * Chainlit's Message.language field is reserved for code highlighting in 2.11.1;
 * it is not rendered as a DOM language/direction attribute for prose messages.
 * This supported custom-JS extension applies direction only to assistant output.
 * It never changes the message's text or its citation metadata.
 */
(() => {
  "use strict";

  const ARABIC = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/u;
  const PAGE_CITATION = /\(page\s+\d+\)/g;
  const ASSISTANT_CONTENT = '[data-step-type="assistant_message"] .message-content';

  function isolateCitations(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (!node.parentElement?.closest("pre, code, .rtl-citation")) {
        textNodes.push(node);
      }
    }

    for (const node of textNodes) {
      const text = node.nodeValue;
      PAGE_CITATION.lastIndex = 0;
      if (!text || !PAGE_CITATION.test(text)) continue;
      PAGE_CITATION.lastIndex = 0;
      const fragment = document.createDocumentFragment();
      let cursor = 0;
      for (const match of text.matchAll(PAGE_CITATION)) {
        fragment.append(text.slice(cursor, match.index));
        const citation = document.createElement("bdi");
        citation.className = "rtl-citation";
        citation.dir = "ltr";
        citation.textContent = match[0];
        fragment.append(citation);
        cursor = match.index + match[0].length;
      }
      fragment.append(text.slice(cursor));
      node.replaceWith(fragment);
    }
  }

  function formatAssistantMessage(content) {
    const isArabic = ARABIC.test(content.textContent || "");
    if (!isArabic) {
      content.classList.remove("rtl-assistant-message");
      content.removeAttribute("dir");
      content.removeAttribute("lang");
      return;
    }

    content.classList.add("rtl-assistant-message");
    content.dir = "rtl";
    content.lang = "ar";
    isolateCitations(content);
  }

  function refresh(root = document) {
    if (root instanceof Element) {
      const message = root.matches(ASSISTANT_CONTENT)
        ? root
        : root.closest(ASSISTANT_CONTENT);
      if (message) formatAssistantMessage(message);
    }
    root.querySelectorAll?.(ASSISTANT_CONTENT).forEach(formatAssistantMessage);
  }

  refresh();
  new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      const root = mutation.target.nodeType === Node.ELEMENT_NODE
        ? mutation.target
        : mutation.target.parentElement;
      refresh(root || document);
    }
  }).observe(document.body, { childList: true, characterData: true, subtree: true });
})();
