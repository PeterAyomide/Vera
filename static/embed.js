/**
 * Vera — Website Lead Capture Embed
 *
 * Drop this script on any page with a contact form.
 * When the form submits, it silently sends the lead to Vera
 * for AI scoring. The form still works normally for the visitor.
 *
 * SETUP:
 *   1. Replace VERA_API_URL with your deployed Vera URL
 *   2. Replace VERA_API_KEY with your API key (from .env → API_KEY)
 *   3. Map your form's field names in FIELD_MAP below
 *   4. Paste this script just before </body> on your contact page
 *
 * That's it. Leads will appear in your Vera dashboard scored and ready.
 */

(function () {
  // ── Configuration ────────────────────────────────────────────────────────
  var VERA_API_URL = "https://your-vera-url.com";   // ← your deployed URL
  var VERA_API_KEY = "your-api-key-here";            // ← your API_KEY from .env

  // Map YOUR form field names to Vera's fields.
  // Change the values (right side) to match your form's input names/ids.
  var FIELD_MAP = {
    name:    "name",         // <input name="name"> or id="name"
    email:   "email",        // <input name="email">
    company: "company",      // <input name="company"> — optional
    message: "message",      // <textarea name="message">
  };

  // ── Lead capture ─────────────────────────────────────────────────────────
  function getField(form, key) {
    var name = FIELD_MAP[key];
    if (!name) return "";
    var el = form.querySelector('[name="' + name + '"], #' + name);
    return el ? (el.value || "").trim() : "";
  }

  function sendToVera(leadData) {
    // Fire and forget — visitor never waits for this
    fetch(VERA_API_URL + "/analyze-lead", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": VERA_API_KEY,
      },
      body: JSON.stringify(leadData),
      // keepalive ensures the request completes even if the page navigates away
      keepalive: true,
    }).catch(function () {
      // Silent fail — never interrupt the visitor's form submission
    });
  }

  function attachToForms() {
    // Intercept all forms on the page
    var forms = document.querySelectorAll("form");
    forms.forEach(function (form) {
      form.addEventListener("submit", function () {
        var name = getField(form, "name");
        var message = getField(form, "message");

        // Only send if we have at least a name and a message
        if (!name || !message) return;

        sendToVera({
          lead_id:  "web-" + Date.now() + "-" + Math.random().toString(36).slice(2, 7),
          name:     name,
          email:    getField(form, "email"),
          company:  getField(form, "company"),
          message:  message,
          source:   "website",
        });
      });
    });
  }

  // Run after DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", attachToForms);
  } else {
    attachToForms();
  }
})();
